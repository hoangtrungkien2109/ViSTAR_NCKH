import grpc
import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
from loguru import logger
from threading import Thread
import numpy as np
import time
import os
from src.ai.services.frame2video_services.HandleConcatFrame import HandleConcatFrame
from src.ai.services.frame2video_services.ser_resources import handle_concat_frame
DELAY_TIME = float(os.getenv("DELAY_TIME"))

def numpy_to_matrix_list(array):
    matrix_list = streaming_pb2.MatrixList()
    for matrix in array:  
        matrix_pb = matrix_list.matrix.add() 
        for row in matrix:
            row_pb = matrix_pb.rows.add()
            row_pb.elements.extend(row.tolist())
    
    return matrix_list

def matrix_list_to_numpy(matrix_list):
    numpy_array = np.array([
        [[element for element in row.elements] for row in matrix.rows]
        for matrix in matrix_list.matrix
    ])

    return numpy_array


def send_image_into_streaming(stub, handle_concat_frame: HandleConcatFrame):
    while True:
        data = handle_concat_frame.pop()
        if data is not None:
            data = numpy_to_matrix_list(data)
            push_image_frame_request = streaming_pb2.PushImageRequest()
            push_image_frame_request.image.CopyFrom(data)
            push_image_frame_request.time_stamp = ""
            push_image_frame_response = stub.PushImage(push_image_frame_request)

            logger.success("Sended successfully")
        time.sleep(DELAY_TIME)

                
def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel=channel)
        pop_frame_response = stub.PopFrame(streaming_pb2.PopFrameRequest(time_stamp=""))
        send_image_thread = Thread(target=send_image_into_streaming, args=(stub,handle_concat_frame))
        send_image_thread.start()

        for response in pop_frame_response:
            if response.request_status == "Success":
                # Push into our queue
                frames = matrix_list_to_numpy(response.frame)
                handle_concat_frame.push_into_process_queue(frames=frames)
                
                
if __name__ == "__main__":
    run()
