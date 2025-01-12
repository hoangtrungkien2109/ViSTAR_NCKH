import grpc
import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
from loguru import logger

def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel=channel)
        pop_frame_response = stub.PopFrame(streaming_pb2.PopFrameRequest(time_stamp=""))
        
        for response in pop_frame_response:
            logger.info(f"Poped frame: {response.frame}")
            
            logger.info("Processing frame...")
            logger.info("Convert into image")
            
            # push_image_response = stub.PushImage(streaming_pb2.PushImageRequest(text=image))
            
            # logger.info(push_image_response.request_status)
            
            
            

            
if __name__ == "__main__":
    run()

