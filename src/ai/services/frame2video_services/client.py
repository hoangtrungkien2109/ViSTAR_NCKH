import base64
import grpc
import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
from loguru import logger

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

def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel=channel)
        pop_frame_response = stub.PopFrame(streaming_pb2.PopFrameRequest(time_stamp=""))

        for response in pop_frame_response:
            if response.request_status == "Empty":
                logger.warning("Waiting...")
            else:
                logger.info(f"Poped frame: {response.frame}")
                logger.info(f"Status: {response.request_status}")
                logger.info("Processing frame...")
                if response.request_status == "Success":
                    base64_image = encode_jfif_to_base64("src\\ai\\services\\frame2video_services\\images.jfif")
                    push_image_response = stub.PushImage(streaming_pb2.PushImageRequest(text=base64_image))
                
            
            


if __name__ == "__main__":
    run()
