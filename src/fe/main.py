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
import speech_recognition as sr
import time

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
    
def recognize_and_send():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Đang lắng nghe... Hãy nói gì đó!")
        recognizer.adjust_for_ambient_noise(source)  # Giảm nhiễu nền
        while True:
            try:
                audio = recognizer.listen(source, timeout=5)  # Lắng nghe trong 5s
                text = recognizer.recognize_google(audio, language="vi-VN")
                words = text.split()
                channel = create_grpc_channel()
                stub = streaming_pb2_grpc.StreamingStub(channel)
                for word in words:
                    status, timestamp = push_text(stub, word)
            
                if "Error" not in status:
                    st.success(f"Text pushed successfully! Status: {status}")
                    # Store message in history
                    st.session_state.messages.append({
                        'text': word,
                        'timestamp': timestamp,
                        'status': status
                    })
                else:
                    st.error(status)
                    time.sleep(0.5)
            except sr.WaitTimeoutError:
                st.warning("⏳ Không phát hiện giọng nói, hãy thử lại!")
            except sr.UnknownValueError:
                st.error("❌ Không nhận diện được, hãy thử lại!")
            except sr.RequestError:
                st.error("⚠️ Lỗi kết nối đến Google API!")
            
            time.sleep(1)  # Tránh vòng lặp quá nhanh

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

    # Create tabs for Push and Pop operations
    tab1, tab2 = st.tabs(["Push Text", "Pop Text"])
    
    # Push Text Tab
    with tab1:
        st.header("Push Text")
        if st.button("🎙️Nhận diện"):
            recognize_and_send()
        
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
if __name__ == "__main__":
    main()
