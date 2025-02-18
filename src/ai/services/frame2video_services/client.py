import grpc
import cv2
import numpy as np
from loguru import logger
from threading import Thread
import time
import os
import src.streaming.pb.streaming_pb2 as streaming_pb2
import src.streaming.pb.streaming_pb2_grpc as streaming_pb2_grpc
from src.ai.services.frame2video_services.HandleConcatFrame import HandleConcatFrame
from src.ai.services.frame2video_services.ser_resources import handle_concat_frame
from src.ai.services.utils.transfrom_data import numpy_to_matrix_list, matrix_list_to_numpy

DELAY_TIME = float(os.getenv("DELAY_TIME"))

def encode_numpy_to_image(array, target_height=720, target_width=1080):
    """
    Convert (1,75,3) numpy array to a 720x1080 image
    """
    # Ensure input array has correct shape
    if array.shape != (1, 75, 3):
        raise ValueError(f"Expected shape (1,75,3), got {array.shape}")
    
    # Reshape the array to remove the first dimension and prepare for visualization
    array = array.squeeze(0)  # Shape becomes (75, 3)
    
    # Scale values to 0-255 if they're float
    if array.dtype in [np.float32, np.float64]:
        array = (array * 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)
    
    # Create a blank image of target size
    img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Calculate scaling factors
    cell_height = target_height // 15  # 5 rows
    cell_width = target_width // 15    # 15 columns
    
    # Place each value from the array into the image grid
    for i in range(75):  # 75 total values
        row = i // 15    # Calculate grid position
        col = i % 15
        
        # Calculate pixel positions
        y_start = row * cell_height
        y_end = (row + 1) * cell_height
        x_start = col * cell_width
        x_end = (col + 1) * cell_width
        
        # Fill the cell with the color value
        color = array[i]
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), color.tolist(), -1)
    
    # Add grid lines for visibility
    for i in range(16):
        x = i * cell_width
        cv2.line(img, (x, 0), (x, target_height), (128, 128, 128), 1)
    for i in range(6):
        y = i * cell_height
        cv2.line(img, (0, y), (target_width, y), (128, 128, 128), 1)
    
    # Encode to jpg format
    success, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise ValueError("Failed to encode image")
    
    return encoded_img.tobytes()

def decode_image_to_numpy(image_bytes, output_shape=(1, 75, 3)):
    """
    Convert 720x1080 image back to (1,75,3) numpy array
    """
    # Decode image bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    
    height, width = img.shape[:2]
    cell_height = height // 15
    cell_width = width // 15
    
    # Extract values from the grid
    output_array = np.zeros((75, 3), dtype=np.uint8)
    for i in range(75):
        row = i // 15
        col = i % 15
        
        # Calculate center of the cell
        y = row * cell_height + cell_height // 2
        x = col * cell_width + cell_width // 2
        
        # Get color value from the center of the cell
        output_array[i] = img[y, x]
    
    # Reshape to final shape
    output_array = output_array.reshape(output_shape)
    
    return output_array

def send_image_into_streaming(stub, handle_concat_frame: HandleConcatFrame):
    while True:
        data = handle_concat_frame.pop()
        if data is not None:
            try:
                # Encode numpy array to image bytes
                image_bytes = encode_numpy_to_image(data)
                
                push_image_response = stub.PushImage(streaming_pb2.PushImageRequest(image=image_bytes))
                
                logger.success(len(handle_concat_frame.processed_frame_queue))
            except Exception as e:
                logger.error(f"Error sending image: {str(e)}")
                
        time.sleep(DELAY_TIME)

def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = streaming_pb2_grpc.StreamingStub(channel=channel)

        pop_frame_response = stub.PopFrame(streaming_pb2.PopFrameRequest(time_stamp=""))
        
        # Start frame processing thread
        send_image_thread = Thread(
            target=send_image_into_streaming, 
            args=(stub, handle_concat_frame),
        )
        send_image_thread.start()
    
        
        for response in pop_frame_response:
            if response.request_status == "Success":
                try:
                    frames = matrix_list_to_numpy(response.frame)
                    handle_concat_frame.push_into_process_queue(frames=frames)
                except Exception as e:
                    logger.error(f"Error processing received frame: {str(e)}")

if __name__ == "__main__":
    run()