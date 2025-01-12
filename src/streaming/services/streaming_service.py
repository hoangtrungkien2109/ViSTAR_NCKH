import threading
from queue import Queue
from loguru import logger
import os

from dotenv import load_dotenv
from grpc import StatusCode
from grpc_interceptor.exceptions import GrpcException

import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
import time

load_dotenv(override=True)
DELAY_TIME = float(os.getenv("DELAY_TIME"))

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
                yield streaming_pb2.PopTextResponse(request_status="Empty", text=None)
            else:
                yield streaming_pb2.PopTextResponse(request_status="Success", text=self.text_queue.get())
            time.sleep(DELAY_TIME)

    def PushFrame(self, request, context):
        try:
            self.frame_queue.put(request.frame)
            return streaming_pb2.PushFrameResponse(request_status="Success")
        except Exception as e:
            raise GrpcException(status_code=StatusCode.INTERNAL, details=str(e)) from e

    def PopFrame(self, request, context):
        while True:
            if self.frame_queue.empty():
                # frame_matrix = streaming_pb2.Matrix(
                #     rows=[
                #         streaming_pb2.MatrixRow(elements=[10.0, 20.0]),  # First row
                #         streaming_pb2.MatrixRow(elements=[30.0, 40.0])   # Second row
                #     ]
                # )
                yield streaming_pb2.PopFrameResponse(request_status="Empty", frame=None)
            else:
                frame = self.frame_queue.get()
                yield streaming_pb2.PopFrameResponse(request_status="Success", frame=frame)
            time.sleep(DELAY_TIME)

    def PushImage(self, request, context):
        try:
            self.image_queue.put(request.text)
            return streaming_pb2.PushImageResponse(request_status="Success")
        except Exception as e:
            raise GrpcException(status_code=StatusCode.INTERNAL, details=str(e)) from e

    def PopImage(self, request, context):
        while True:
            if self.image_queue.empty():
                yield streaming_pb2.PopImageResponse(request_status="Empty", text=None)
            else:
                image = self.image_queue.get()
                yield streaming_pb2.PopImageResponse(request_status="Success", text=image)
            time.sleep(DELAY_TIME)
