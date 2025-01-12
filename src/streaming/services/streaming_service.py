import threading
from queue import Queue

from grpc import StatusCode
from grpc_interceptor.exceptions import GrpcException

import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc


class StreamingBaseService(streaming_pb2_grpc.StreamingServicer):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(StreamingBaseService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True

            self.text_queue = Queue()
            self.frame_queue = Queue()
            self.image_queue = Queue()

    def PushText(self, request, context):
        try:
            self.text_queue.put(request.text)
            return streaming_pb2.PushTextResponse(request_status="Success")
        except Exception as e:
            raise GrpcException(status_code=StatusCode.INTERNAL, details=str(e)) from e

    def PopText(self, request, context):
        if self.text_queue.empty():
            raise GrpcException(status_code=StatusCode.NOT_FOUND, details="No text available in the queue.")
        text = self.text_queue.get()
        return streaming_pb2.PopTextResponse(request_status="Success", text=text)

    def PushFrame(self, request, context):
        try:
            self.frame_queue.put(request.frame)
            return streaming_pb2.PushFrameResponse(request_status="Success")
        except Exception as e:
            raise GrpcException(status_code=StatusCode.INTERNAL, details=str(e)) from e

    def PopFrame(self, request, context):
        if self.frame_queue.empty():
            raise GrpcException(status_code=StatusCode.NOT_FOUND, details="No frame available in the queue.")
        frame = self.frame_queue.get()
        return streaming_pb2.PopFrameResponse(request_status="Success", frame=frame)

    def PushImage(self, request, context):
        try:
            self.image_queue.put(request.image_base64)
            return streaming_pb2.PushImageResponse(request_status="Success")
        except Exception as e:
            raise GrpcException(status_code=StatusCode.INTERNAL, details=str(e)) from e

    def PopImage(self, request, context):
        if self.image_queue.empty():
            raise GrpcException(status_code=StatusCode.NOT_FOUND, details="No image available in the queue.")
        image = self.image_queue.get()
        return streaming_pb2.PopImageResponse(request_status="Success", text=image)
