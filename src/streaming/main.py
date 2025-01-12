import logging
from concurrent import futures

import grpc
# from grpc_interceptor import ExceptionToStatusInterceptor

from src.streaming.pb.streaming_pb2_grpc import add_StreamingServicer_to_server
from src.streaming.services.streaming_service import StreamingBaseService


class StreamingService(StreamingBaseService):
    pass

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))  # Create gRPC server
    add_StreamingServicer_to_server(StreamingService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server is running")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()