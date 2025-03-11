import grpc
import cv2
import numpy as np
from loguru import logger
from threading import Thread
import os
import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
from src.ai.services.frame2video_services.HandleConcatFrame import HandleConcatFrame
from src.ai.services.frame2video_services.ser_resources import handle_concat_frame
from src.ai.services.utils.transfrom_data import matrix_list_to_numpy
import cv2
import numpy as np

DELAY_TIME = float(os.getenv("DELAY_TIME"))


def visualize_landmarks_minimal(array, target_height=480, target_width=720, line_thickness=1):
    if array.shape != (1, 75, 3):
        raise ValueError(f"Expected shape (1,75,3), got {array.shape}")
    
    img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    points = (array.squeeze(0)[:, :2] * [target_width, target_height]).astype(np.float32)
    
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12),
        (11, 13), (13, 15), (15, 17),
        (12, 14), (14, 16), (16, 18),
        (23, 24),
        (24, 26), (26, 28), (28, 32),
        (23, 25), (25, 27), (27, 29), (29, 31),
    ]
    
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20)
    ]
    
    pose_landmarks = points[:33]
    right_hand_landmarks = points[33:54]
    left_hand_landmarks = points[54:]
    
    for lm in pose_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]):
            x, y = int(lm[0]), int(lm[1])
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
    
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = pose_landmarks[start_idx]
        end_point = pose_landmarks[end_idx]
        if not np.isnan(start_point[0]) and not np.isnan(end_point[0]):
            start_x, start_y = int(start_point[0]), int(start_point[1])
            end_x, end_y = int(end_point[0]), int(end_point[1])
            cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), line_thickness)
    
    for lm in right_hand_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]):
            x, y = int(lm[0]), int(lm[1])
            cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
    
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = right_hand_landmarks[start_idx]
        end_point = right_hand_landmarks[end_idx]
        if not np.isnan(start_point[0]) and not np.isnan(end_point[0]):
            start_x, start_y = int(start_point[0]), int(start_point[1])
            end_x, end_y = int(end_point[0]), int(end_point[1])
            cv2.line(img, (start_x, start_y), (end_x, end_y), (255, 0, 0), line_thickness)
    
    for lm in left_hand_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]):
            x, y = int(lm[0]), int(lm[1])
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
    
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = left_hand_landmarks[start_idx]
        end_point = left_hand_landmarks[end_idx]
        if not np.isnan(start_point[0]) and not np.isnan(end_point[0]):
            start_x, start_y = int(start_point[0]), int(start_point[1])
            end_x, end_y = int(end_point[0]), int(end_point[1])
            cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 0, 255), line_thickness)
    
    success, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not success:
        raise ValueError("Failed to encode image")
    
    return encoded_img.tobytes()

def send_image_into_streaming(stub, handle_concat_frame: HandleConcatFrame):
    mem = None
    while True:       
        data = handle_concat_frame.pop()
        
        if is_similar_frame(frame1=mem, frame2=data) and handle_concat_frame.getLen() > 100:
            continue
        else:
            pass
        if data is not None:
            try:
                # Encode numpy array to image bytes
                image_bytes = visualize_landmarks_minimal(data)
                
                push_image_response = stub.PushImage(streaming_pb2.PushImageRequest(image=image_bytes))
                
                logger.success(len(handle_concat_frame.processed_frame_queue))
            except Exception as e:
                logger.error(f"Error sending image: {str(e)}")
            mem = data
        else:
            mem = None

def is_similar_frame(frame1, frame2, threshold=0.1):
    if frame1 is None or frame2 is None:
        return False
    distance = np.linalg.norm(frame1 - frame2)
    return distance < threshold

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