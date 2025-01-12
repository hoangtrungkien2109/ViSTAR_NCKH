import grpc
import streaming.pb.streaming_pb2 as streaming_pb2
import streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
import time


def push_text():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel=channel)
        response = stub.PushText(streaming_pb2.PushTextRequest(text="Xin chao moi nguoi"))
        
        print("Push text status", response)
        
def pop_text():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel=channel)
        response = stub.PopText(streaming_pb2.PopTextRequest(time_stamp=time.time))
        ...
        response = stub.PushFrame(stre)
        print("Pop text from server", response)
        
