import grpc
import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
from loguru import logger
import numpy as np

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

def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel=channel)
        pop_text_response = stub.PopText(streaming_pb2.PopTextRequest(time_stamp=""))

        for response in pop_text_response:
            if response.request_status == "Success":
                
                logger.info(response.text)
                array = np.random.rand(3, 4, 5).astype(np.float32)
                
                # Convert numpy array to MatrixList
                matrix_list = numpy_to_matrix_list(array)

                if response.text == "add frame":
                    push_frame_response = stub.PushFrame(streaming_pb2.PushFrameRequest(frame=matrix_list))
                
        
            
if __name__ == "__main__":
    run()
