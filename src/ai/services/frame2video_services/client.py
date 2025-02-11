import base64
import grpc
import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
from loguru import logger
from threading import Thread
import numpy as np

from src.ai.services.frame2video_services.HandleConcatFrame import HandleConcatFrame
from src.ai.services.frame2video_services.ser_resources import handle_concat_frame

def numpy_to_matrix_list(array):
    matrix_list = streaming_pb2.MatrixList()
    for matrix in array:
        matrix_row = streaming_pb2.Matrix()
        for row in matrix:
            matrix_row.rows.add(elements=row)
        matrix_list.matrix.append(matrix_row)
    return matrix_list

def matrix_list_to_numpy(matrix_list):
    return np.array([[[element for element in row.elements] for row in matrix.rows] for matrix in matrix_list.matrix])

def encode_jfif_to_base64(file_path):
    try:
        # Open the JFIF file in binary mode
        with open(file_path, "rb") as image_file:
            # Read the file and encode it to base64
            base64_string = base64.b64encode(image_file.read()).decode('utf-8')
        return base64_string
    except Exception as e:
        print(f"Error encoding file to base64: {e}")
        return None

def send_image_into_streaming(stub, handle_concat_frame: HandleConcatFrame):
    while True:
        data = handle_concat_frame.pop()
        if data:
            base64_image = encode_jfif_to_base64("src/ai/services/frame2video_services/images.jfif")
            push_image_response = stub.PushImage(streaming_pb2.PushImageRequest(text=base64_image))
            logger.success("Sended successfully")
                
def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel=channel)
        pop_frame_response = stub.PopFrame(streaming_pb2.PopFrameRequest(time_stamp=""))
        # send_image_thread = Thread(target=send_image_into_streaming, args=(stub,handle_concat_frame))
        # send_image_thread.start()    
            
        for response in pop_frame_response:
            # Push into our queue
            logger.info(response.request_status)
            frames = matrix_list_to_numpy(response.frame)
            
            handle_concat_frame.push_into_process_queue(frames=frames)
                
                
if __name__ == "__main__":
    run()
