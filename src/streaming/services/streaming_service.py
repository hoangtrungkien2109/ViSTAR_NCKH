import threading
from queue import Queue
from loguru import logger
from grpc import StatusCode
from grpc_interceptor.exceptions import GrpcException

import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
import time

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
            logger.info(request.text)
            self.text_queue.put(request.text)
            return streaming_pb2.PushTextResponse(request_status="Success")
        except Exception as e:
            raise GrpcException(status_code=StatusCode.INTERNAL, details=str(e)) from e

    def PopText(self, request, context):
        while True:
            if self.text_queue.empty():
                yield streaming_pb2.PopTextResponse(request_status="Success", text="Empty")
            else:
                yield streaming_pb2.PopTextResponse(request_status="Success", text=self.text_queue.get())
            time.sleep(1)

    def PushFrame(self, request, context):
        try:
            self.frame_queue.put(request.frame)
            return streaming_pb2.PushFrameResponse(request_status="Success")
        except Exception as e:
            raise GrpcException(status_code=StatusCode.INTERNAL, details=str(e)) from e

    def PopFrame(self, request, context):
        while True:
            if self.frame_queue.empty():
                # raise GrpcException(status_code=StatusCode.NOT_FOUND, details="No frame available in the queue.")
                
                yield streaming_pb2.PopFrameResponse(request_status="Success", frame=None)
            else:
                frame = self.frame_queue.get()
                yield streaming_pb2.PopFrameResponse(request_status="Success", frame=frame)
            time.sleep(1)
            

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
