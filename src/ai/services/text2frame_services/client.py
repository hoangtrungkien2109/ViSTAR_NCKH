import grpc
from loguru import logger
import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
from src.ai.services.text2frame_services.mapping_service import SimilaritySentence

def run():
    ss: SimilaritySentence = SimilaritySentence(
        default_dict_path="D:\\NCKH\\Text_to_Sign\\ViSTAR\src\\ai\\services\\text2frame_services\\data\\character_dict.json"
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
                if response.text == "add frame":
                    push_frame_response = stub.PushFrame(streaming_pb2.PushFrameRequest(frame=frame_matrix_list))
                    logger.info(push_frame_response.request_status)



if __name__ == "__main__":
    run()
