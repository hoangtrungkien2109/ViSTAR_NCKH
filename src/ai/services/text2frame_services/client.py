import grpc
import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
from loguru import logger
import numpy as np

def numpy_to_matrix_list(array):
    matrix_list = streaming_pb2.MatrixList()
    
    print(f"Converting numpy array of shape {array.shape} to MatrixList...")

    for matrix in array:  # Loop over batch dimension (N)
        matrix_pb = matrix_list.matrix.add()  # Add a new matrix to the list
        for row in matrix:  # Loop over rows
            row_pb = matrix_pb.rows.add()  # Add a new row
            row_pb.elements.extend(row.tolist())  # Convert row to list and extend
    
    return matrix_list

def matrix_list_to_numpy(matrix_list):
    numpy_array = np.array([
        [[element for element in row.elements] for row in matrix.rows]
        for matrix in matrix_list.matrix
    ])

    print(f"Converted back to numpy array of shape {numpy_array.shape}")
    return numpy_array

def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel=channel)
        pop_text_response = stub.PopText(streaming_pb2.PopTextRequest(time_stamp=""))

        for response in pop_text_response:
            if response.request_status == "Success":
                
                logger.info(response.text)
                
                if response.text == "1":
                    array = np.random.rand(5, 75, 3).astype(np.float32)
                    # Convert numpy array to MatrixList
                    matrix_list = numpy_to_matrix_list(array)
                    print(matrix_list)
                    push_frame_response = stub.PushFrame(streaming_pb2.PushFrameRequest(frame=matrix_list))
                
        
            
if __name__ == "__main__":
    run()
