import streamlit as st
import grpc
import numpy as np
import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
import cv2
import time
from PIL import Image

def matrix_list_to_numpy(matrix_list):
    return np.array([[[element for element in row.elements] for row in matrix.rows] for matrix in matrix_list.matrix])

def fetch_frame():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel)
        pop_frame_response = stub.PopFrame(streaming_pb2.PopFrameRequest(time_stamp=""))
        
        for response in pop_frame_response:
            frames = matrix_list_to_numpy(response.frame)
            if len(frames) > 0:
                frame = frames[0]  # Assuming the first frame
                return frame
    return None

def main():
    st.title("Real-time Frame Visualization")
    frame_placeholder = st.empty()
    
    while True:
        frame = fetch_frame()
        if frame is not None:
            frame = (frame * 255).astype(np.uint8)  # Normalize if needed
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) if len(frame.shape) == 2 else frame
            image = Image.fromarray(frame)
            frame_placeholder.image(image, channels="RGB")
        time.sleep(0.1)

if __name__ == "__main__":
    main()
