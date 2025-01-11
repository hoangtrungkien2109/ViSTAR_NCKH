import shutil
import os
from time import time
import cv2
import ffmpeg
import numpy as np
from loguru import logger
from src.ai.services.frame2video_services.lstm_model import load_model, predict

# Get the base path dynamically
base_path = os.path.dirname(os.path.abspath(__file__))  # This gives the directory of the current script

# Construct the full path to the model
# MODEL_PATH = os.path.join(base_path, 'cut', 'cut.pth')
MODEL_PATH = "src/models/model_utils/manipulate/cut/cut.pth"

VIDEO_PATH = "src/web/static"


# Correct connections for pose landmarks in MediaPipe (total 33 landmarks)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),          # Right eye to ear
    (0, 4), (4, 5), (5, 6), (6, 8),          # Left eye to ear
    (9, 10), (11, 12),                       # Shoulders
    (11, 13), (13, 15), (15, 17),            # Left arm
    (12, 14), (14, 16), (16, 18),            # Right arm
    (23, 24),                                # Hips
    (24, 26), (26, 28), (28, 32),            # Right leg
    (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (5, 6), (6, 7), (7, 8),          # Index finger
    (9, 10), (10, 11), (11, 12),     # Middle finger
    (13, 14), (14, 15), (15, 16),    # Ring finger
    (17, 18), (18, 19), (19, 20)     # Pinky finger
]

def draw_landmarks(image, frame_landmarks, line_thickness=2):
    pose_landmarks = frame_landmarks[:33]
    for lm in pose_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]) and not np.isnan(lm[2]):  # Check for NaN values
            x, y = int(lm[0] * image.shape[1]), int(lm[1] * image.shape[0])
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

    # Draw lines between connected pose landmarks
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = pose_landmarks[start_idx]
        end_point = pose_landmarks[end_idx]
        if (start_point[0] != 0 and start_point[1] != 0) and (end_point[0] != 0 and end_point[1] != 0):
            start_x, start_y = int(start_point[0] * image.shape[1]), int(start_point[1] * image.shape[0])
            end_x, end_y = int(end_point[0] * image.shape[1]), int(end_point[1] * image.shape[0])
            cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), line_thickness)

    right_hand_landmarks = frame_landmarks[33:33 + 21]
    right_hand_present = False
    for lm in right_hand_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]) and not np.isnan(lm[2]):  # Check for NaN values
            x, y = int(lm[0] * image.shape[1]), int(lm[1] * image.shape[0])
            cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
            right_hand_present = True
            
    # Draw lines between connected right hand landmarks
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = right_hand_landmarks[start_idx]
        end_point = right_hand_landmarks[end_idx]
        if (start_point[0] != 0 and start_point[1] != 0) and (end_point[0] != 0 and end_point[1] != 0):
            start_x, start_y = int(start_point[0] * image.shape[1]), int(start_point[1] * image.shape[0])
            end_x, end_y = int(end_point[0] * image.shape[1]), int(end_point[1] * image.shape[0])
            cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), line_thickness)
            
    left_hand_landmarks = frame_landmarks[33 + 21:]
    left_hand_present = False
    for lm in left_hand_landmarks:
        if not np.isnan(lm[0]) and not np.isnan(lm[1]) and not np.isnan(lm[2]):  # Check for NaN values
            x, y = int(lm[0] * image.shape[1]), int(lm[1] * image.shape[0])
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
            left_hand_present = True
            
    # Draw lines between connected left hand landmarks
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = left_hand_landmarks[start_idx]
        end_point = left_hand_landmarks[end_idx]
        if (start_point[0] != 0 and start_point[1] != 0) and (end_point[0] != 0 and end_point[1] != 0):
            start_x, start_y = int(start_point[0] * image.shape[1]), int(start_point[1] * image.shape[0])
            end_x, end_y = int(end_point[0] * image.shape[1]), int(end_point[1] * image.shape[0])
            cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), line_thickness)
             
    if not right_hand_present and not left_hand_present:
        return None  # Return None if no hands are detected

    return image
def load_and_concatenate_npy_files(model, list_landmarks_data):
    all_landmarks = []
    for idx, landmarks_data in enumerate(list_landmarks_data):
        # logger.info(npy_file) 
        # landmarks_data = np.load(npy_file)
        # logger.debug(landmarks_data.shape)
        # logger.error(len(list_landmarks_data))
        # logger.error(len(list_landmarks_data[0]))
        # logger.error(len(list_landmarks_data[0][0]))
        # logger.error(len(list_landmarks_data[0][0][0]))
        # logger.error(len(list_landmarks_data[1]))
        landmarks_data = np.array(landmarks_data)
        # landmarks_data = landmarks_data[~np.any(landmarks_data == 0, axis=(1,2))]

        # logger.info(landmarks_data.shape)

        if len(landmarks_data) >= 300:
            landmarks_data = landmarks_data[:300]

        if len(landmarks_data) == 0:
            logger.error("CO")
            continue
        logger.info(landmarks_data.shape)
        
        p = predict(model,landmarks_data)

        landmarks_data = landmarks_data[p.flatten() == 1]

        if len(landmarks_data) == 0:
            continue
        
        if len(all_landmarks) == 0:
            all_landmarks.append(landmarks_data)
        else:
            # logger.info("Hello")
            logger.info(np.linalg.norm(all_landmarks[-1][-1] - landmarks_data[0]))
            if np.linalg.norm(all_landmarks[-1][-1] - landmarks_data[0]) <= 1:
                middle = np.linspace(all_landmarks[-1][-1], landmarks_data[0], num=7)
            elif np.linalg.norm(all_landmarks[-1][-1] - landmarks_data[0]) <= 2:
                middle = np.linspace(all_landmarks[-1][-1], landmarks_data[0], num=10)
            else:
                middle = np.linspace(all_landmarks[-1][-1], landmarks_data[0], num=15)
            all_landmarks.append(middle)
            all_landmarks.append(landmarks_data)
        
    concatenated_landmarks = np.concatenate(all_landmarks,axis=0)
    return concatenated_landmarks

def is_similar_frame(frame1, frame2, threshold=0.05):
    if frame1 is None:
        return False
    distance = np.linalg.norm(frame1 - frame2)
    return distance < threshold

def defineSE(arr):
    s, e = 0, len(arr) - 1
    for i in range(len(arr) - 1):
        if arr[i] != arr[i + 1]:
            s = i + 1
            break
            
    for i in range(len(arr) - 1, 0, -1):
        if arr[i] != arr[i - 1]:
            e = i
            break
    return s, e
    


# load model
model = load_model(MODEL_PATH)

# npy_folder = './temp'
# npy_files = glob.glob(os.path.join(npy_folder, '*.npy'))

# concatenated_landmarks_array = load_and_concatenate_npy_files(model, npy_files)

# frame_index = 0
# num_frames = len(concatenated_landmarks_array)

def save_frames_to_output(landmarks_array, return_format='video', fps = 30):
    # logger.warning(landmarks_array.shape)
    concatenated_landmarks_array = load_and_concatenate_npy_files(model, landmarks_array)
    frame_index = 0
    num_frames = len(concatenated_landmarks_array)
    logger.warning(f"Num frame: {num_frames}")
    image_height, image_width = 720, 1280
    frame_index = 0

    # Initialize a list to store the frames
    frames = []

    # start, end = defineSE(concatenated_prediction_array[:, 0])
    last_frame_landmarks = None

    while frame_index < num_frames:
        image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
        frame_landmarks = concatenated_landmarks_array[frame_index]

        result_image = draw_landmarks(image, frame_landmarks)
        # if (frame_index >= start and frame_index <= end) and concatenated_prediction_array[frame_index][0] == 0:
        #     frame_index += 1
        #     continue

        if result_image is None or is_similar_frame(last_frame_landmarks, frame_landmarks):
            frame_index += 1
            continue

        last_frame_landmarks = frame_landmarks

        # Append the frame to the list
        frames.append(result_image)
        
        frame_index += 1
        # Append the frame to the list
        frames.append(result_image)
        
        frame_index += 1


    # Use ffmpeg to create a video from the frames
    if frames:
        name = int(time())
        out_file = f"{VIDEO_PATH}/{name}.mp4"
        # Convert list of frames to a numpy array (height, width, channels, num_frames)
        frames_array = np.array(frames)
        
        # Create a video using ffmpeg
        if os.path.exists(out_file):
            shutil.rmtree(out_file)

        ffmpeg.input('pipe:0', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(image_width, image_height), r=fps).output(out_file, vcodec='libx264').run(input=frames_array.tobytes())
        print(f"Video saved to {out_file}")

    if return_format == 'video':
        return f'{name}.mp4'  # Path to the video file
    else:
        raise ValueError("Invalid return format. Choose 'npy' or 'video'.")



if __name__ == '__main__':
    # Example usage
    # output_file = save_frames_to_output(concatenated_landmarks_array, num_frames, return_format='video')
    # print("Output file:", output_file)
    pass