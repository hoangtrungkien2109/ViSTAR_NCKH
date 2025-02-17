import streamlit as st
import grpc
import datetime
import numpy as np
import cv2
import time
from PIL import Image
import streaming_pb2 as streaming_pb2
import streaming_pb2_grpc as streaming_pb2_grpc
from loguru import logger
import os
DELAY_TIME = float(os.getenv("DELAY_TIME"))

# Constants for visualization
IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1024

def create_grpc_channel():
    """Creates and returns a gRPC channel."""
    return grpc.insecure_channel("localhost:50051")

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

def push_text(stub, text):
    """Sends text to the gRPC server."""
    try:
        timestamp = get_timestamp()
        request = streaming_pb2.PushTextRequest(text=text, time_stamp=timestamp)
        response = stub.PushText(request)
        return response.request_status, timestamp
    except grpc.RpcError as e:
        return f"Error: {str(e)}", None

def matrix_list_to_numpy(matrix_list):
    """Converts a gRPC matrix list to a NumPy array."""
    return np.array([[[element for element in row.elements] for row in matrix.rows] for matrix in matrix_list.matrix])

def fetch_image_frame(stub):
    """Fetches image frame data from the gRPC server."""
    try:
        pop_image_frame_response = stub.PopImage(streaming_pb2.PopImageRequest(time_stamp=""))
        for response in pop_image_frame_response:
            images = matrix_list_to_numpy(response.image)
            return images
    except grpc.RpcError as e:
        st.error(f"gRPC Error: {e}")
        return None

def visualize_landmarks(frame):
    """
    Takes a frame of shape (1, 75, 3) and visualizes it on a blank 720x1024 canvas.
    """
    if frame.shape != (1, 75, 3):
        return None

    frame = frame[0]  # Remove the first dimension to get shape (75, 3)

    # Create a blank black image
    blank_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    for x, y, _ in frame:
        # Convert normalized coordinates (assuming 0-1 range) to pixel coordinates
        px, py = int(x * IMAGE_WIDTH), int(y * IMAGE_HEIGHT)

        # Ensure the points are within bounds
        if 0 <= px < IMAGE_WIDTH and 0 <= py < IMAGE_HEIGHT:
            cv2.circle(blank_image, (px, py), 5, (0, 255, 0), -1)  # Draw green circle

    return blank_image

def main():
    st.title("Streaming Text Service")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'running' not in st.session_state:
        st.session_state.running = False

    # Create gRPC channel and stub
    try:
        channel = create_grpc_channel()
        stub = streaming_pb2_grpc.StreamingStub(channel)
        st.sidebar.success("Connected to gRPC server")
    except Exception as e:
        st.sidebar.error(f"Failed to connect: {str(e)}")
        return

    # Push Text Section
    st.header("Push Text")
    text_input = st.text_area("Enter your text:", height=100)

    if st.button("Send Text"):
        if text_input.strip():
            status, timestamp = push_text(stub, text_input)
            if "Error" not in status:
                st.success(f"Text pushed successfully! Status: {status}")
                st.session_state.messages.append({'text': text_input, 'timestamp': timestamp, 'status': status})
            else:
                st.error(status)
        else:
            st.warning("Please enter some text before sending.")

    # Real-time Frame Visualization
    st.header("Real-time Frame Visualization")

    if st.button("Start Streaming"):
        st.session_state.running = True
    if st.button("Stop Streaming"):
        st.session_state.running = False

    frame_placeholder = st.empty()

    while st.session_state.running:
        frame = fetch_image_frame(stub)
        logger.info("Visualize")
        if frame is not None:
            visualized_frame = visualize_landmarks(frame)
            if visualized_frame is not None:
                logger.info("Visualize")
                frame_placeholder.image(visualized_frame, channels="RGB")
        time.sleep(DELAY_TIME)

        # st.rerun()


if __name__ == "__main__":
    main()
