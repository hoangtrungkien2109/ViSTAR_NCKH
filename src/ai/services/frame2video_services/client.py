import grpc
import cv2
import numpy as np
from loguru import logger
from threading import Thread
import time
import os
import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
from src.ai.services.frame2video_services.HandleConcatFrame import HandleConcatFrame
from src.ai.services.frame2video_services.ser_resources import handle_concat_frame
from src.ai.services.utils.transfrom_data import numpy_to_matrix_list, matrix_list_to_numpy

DELAY_TIME = float(os.getenv("DELAY_TIME"))

def visualize_landmarks_minimal(array, target_height=480, target_width=720):

    if array.shape != (1, 75, 3):
        raise ValueError(f"Expected shape (1,75,3), got {array.shape}")
    
    img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    points = (array.squeeze(0)[:, :2] * [target_width, target_height]).astype(np.int32)
    
    for point in points:
        cv2.circle(img, tuple(point), 1, (0, 255, 0), -1)
    
    success, encoded_img = cv2.imencode('.jpg', img, 
                                      [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not success:
        raise ValueError("Failed to encode image")
    
    return encoded_img.tobytes()

def send_image_into_streaming(stub, handle_concat_frame: HandleConcatFrame):
    while True:
        data = handle_concat_frame.pop()
        if data is not None:
            try:
                # Encode numpy array to image bytes
                image_bytes = visualize_landmarks_minimal(data)
                
                push_image_response = stub.PushImage(streaming_pb2.PushImageRequest(image=image_bytes))
                
                logger.success(len(handle_concat_frame.processed_frame_queue))
            except Exception as e:
                logger.error(f"Error sending image: {str(e)}")
                
        # time.sleep(DELAY_TIME)

def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel=channel)

        pop_frame_response = stub.PopFrame(streaming_pb2.PopFrameRequest(time_stamp=""))
        
        # Start frame processing thread
        send_image_thread = Thread(
            target=send_image_into_streaming, 
            args=(stub, handle_concat_frame),
        )
        send_image_thread.start()
    
        
        for response in pop_frame_response:
            if response.request_status == "Success":
                try:
                    frames = matrix_list_to_numpy(response.frame)
                    handle_concat_frame.push_into_process_queue(frames=frames)
                except Exception as e:
                    logger.error(f"Error processing received frame: {str(e)}")

if __name__ == "__main__":
    run()