import os
import time
from threading import Thread
import concurrent
from dotenv import load_dotenv
import grpc
from loguru import logger
import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
from src.ai.services.text2frame_services.mapping_service import SimilaritySentence
load_dotenv()

class StreamProcessor:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)  # Adjust workers as needed
        self.channel = grpc.insecure_channel("localhost:50051")
        self.stub = streaming_pb2_grpc.StreamingStub(self.channel)
        self.ss = SimilaritySentence(
            default_dict_path="D:/tnchau/Project/ViSTAR/data/character_dict.json"
        )

    def pop_text_stream(self):
        """Continuously pops words from the gRPC stream and adds them to the queue."""
        pop_text_response = self.stub.PopText(streaming_pb2.PopTextRequest(time_stamp=""))
        for response in pop_text_response:
            if response.request_status == "Empty":
                logger.warning("Waiting...")
            else:
                logger.info(f"Popped text from server: {response.text}")
                logger.info(f"Processing text: {response.text}")
                self.ss.push_word(response.text)

    def process_word(self):
        """Continuously processes words from the queue in parallel."""
        while True:
            frames = self.ss.get_frame()
            logger.info("Converted into frame")
            time.sleep(os.getenv("DELAY_TIME"))
            # Construct Matrix
            frame_matrix_list = streaming_pb2.MatrixList(
                matrix=[
                    streaming_pb2.Matrix(
                        rows=[
                            streaming_pb2.MatrixRow(elements=point)
                            for point in frame
                        ]
                    ) for frame in frames
                ]
            )
            push_frame_response = self.stub.PushFrame(
                streaming_pb2.PushFrameRequest(frame=frame_matrix_list)
            )
            logger.info(push_frame_response.request_status)

    def run(self):
        """Runs both the pop and process functions in parallel."""
        pop_thread = Thread(target=self.pop_text_stream, daemon=True)
        pop_thread.start()

        # Use ThreadPoolExecutor to process words in parallel
        for _ in range(10):  # Number of workers
            self.executor.submit(self.process_word)

        pop_thread.join()
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
    ss: SimilaritySentence = SimilaritySentence(
        default_dict_path=os.getenv("DEFAULT_DICT")
    )
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel=channel)
        pop_text_response = stub.PopText(streaming_pb2.PopTextRequest(time_stamp=""))

        for response in pop_text_response:
            if response.request_status == "Empty":
                logger.warning("Waiting...")
            else:
                logger.info(f"Poped text from server: {response.text}")
                ss.push_word(response.text)
                logger.info("Processing text...")
                frames = ss.get_frame()
                logger.info("Convert into frame")

                # Construct Matrix with rows and elements
                frame_matrix_list = streaming_pb2.MatrixList(
                    matrix=[
                        streaming_pb2.Matrix(
                            rows=[
                                streaming_pb2.MatrixRow(
                                    elements=point
                                ) for point in frame
                            ]
                        ) for frame in frames
                    ]
                )

                # Construct Matrix with rows and elements
                # frame_matrix = streaming_pb2.Matrix(
                #     rows=[
                #         streaming_pb2.MatrixRow(elements=[1.0, 2.0]),  # First row
                #         streaming_pb2.MatrixRow(elements=[3.0, 4.0])   # Second row
                #     ]
                # )
            frames = ss.get_frame()
            logger.info("Convert into frame")

            # Construct Matrix with rows and elements
            frame_matrix_list = streaming_pb2.MatrixList(
                matrix=[
                    streaming_pb2.Matrix(
                        rows=[
                            streaming_pb2.MatrixRow(
                                elements=point
                            ) for point in frame
                        ]
                    ) for frame in frames
                ]
            )
                # if response.text == "add frame":
            push_frame_response = stub.PushFrame(streaming_pb2.PushFrameRequest(frame=frame_matrix_list))
            logger.info(push_frame_response.request_status)
        
            
if __name__ == "__main__":
    # processor = StreamProcessor()
    # processor.run()
    run()