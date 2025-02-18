import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
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
