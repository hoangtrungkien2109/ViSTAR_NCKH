import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
LEFT_HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
RIGHT_HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS




actions = np.array(["Xin Loi","Tu Choi","Xin Chao","Thay","May Man","Cam on","Ky nang","Ruc Ro","Dia chi","Nhan vien","Tiep tan","San truong","Khong quen","Le Halloween","Ngay nay","Nghi hoc","Toi"]) # add ky nang, ruc ro

def mediapipe_detection(image, model):
    """
    Processes an image using a specified Mediapipe model, converting the image to RGB format
    before processing and restoring it to BGR format afterward.

    Parameters
    ----------
    image : numpy.ndarray
        The input image in BGR format.
    model : mediapipe.python.solutions.hands.Hands
        The Mediapipe model used to process the image, typically initialized as a hand
        or pose detection model.

    Returns
    -------
    tuple
        A tuple containing:
        - image (numpy.ndarray): The image restored to BGR format after processing.
        - results (mediapipe.python.solutions.hands.Hands.process): The results object
          containing the processed data from the Mediapipe model, including detected landmarks.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    """
    Draws facial, pose, and hand landmarks on the provided image using MediaPipe drawing utilities.

    This function overlays the detected facial landmarks, pose landmarks, left hand landmarks,
    and right hand landmarks onto the input image. It uses predefined connection styles for each
    type of landmarks.

    Parameters
    ----------
    image : numpy.ndarray
        The image on which landmarks will be drawn. Typically, this is a BGR image as used by OpenCV.

    results : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        The results object obtained from a MediaPipe holistic model that contains detected landmarks.

    Returns
    -------
    None
        The function modifies the input image in place by drawing the landmarks.
    """
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )


def draw_styled_landmarks(image, results):
    """
    Draws styled pose and hand landmarks on the provided image using customized drawing specifications.

    This function overlays the detected pose landmarks, left hand landmarks, and right hand landmarks
    onto the input image with customized colors, thicknesses, and circle radii for better visualization.

    Parameters
    ----------
    image : numpy.ndarray
        The image on which landmarks will be drawn. Typically, this is a BGR image as used by OpenCV.

    results : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        The results object obtained from a MediaPipe holistic model that contains detected landmarks.

    Returns
    -------
    None
        The function modifies the input image in place by drawing the styled landmarks.
    """
    # Draw pose landmarks with custom styles
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )

    # Draw left hand landmarks with custom styles
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )

    # Draw right hand landmarks with custom styles
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )


def compute_distances(landmarks, connections):
    """
    Computes the Euclidean distances between connected landmark points.

    Given a set of landmarks and their connections, this function calculates the Euclidean distance
    between each pair of connected points in 3D space (x, y, z).

    Parameters
    ----------
    landmarks : list of mediapipe.framework.formats.landmark_pb2.NormalizedLandmark
        A list of landmark points detected by MediaPipe. Each landmark contains x, y, z coordinates.

    connections : list of tuple of int
        A list of tuples where each tuple contains two integers representing the indices
        of landmarks that are connected.

    Returns
    -------
    distances : list of float
        A list containing the Euclidean distances between each pair of connected landmarks.
    """
    distances = []
    for connection in connections:
        start_idx, end_idx = connection
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        # Compute Euclidean distance in 3D space
        distance = np.linalg.norm(
            np.array([start_point.x, start_point.y, start_point.z]) -
            np.array([end_point.x, end_point.y, end_point.z])
        )
        distances.append(distance)
    return distances


def extract_keypoints(results):
    """
    Extracts and concatenates keypoints and related features from MediaPipe holistic results.

    This function processes the pose, left hand, and right hand landmarks to extract their
    coordinates and computes the distances between connected landmarks. Additionally, it calculates
    proximity features such as the distance between the wrists and the nose.

    Parameters
    ----------
    results : mediapipe.solutions.holistic.Holistic
        The results object obtained from a MediaPipe holistic model that contains detected landmarks.

    Returns
    -------
    feature_vector : numpy.ndarray
        A 1D NumPy array containing the concatenated keypoints, connection distances, and proximity features.
        The structure is as follows:
            - Pose keypoints (flattened x, y, z, visibility)
            - Left hand keypoints (flattened x, y, z)
            - Right hand keypoints (flattened x, y, z)
            - Pose connection distances
            - Left hand connection distances
            - Right hand connection distances
            - Proximity features (distance between left wrist and nose, distance between right wrist and nose)

    Notes
    -----
    - If certain landmarks are not detected (e.g., pose or hands), the corresponding features are filled with zeros.
    - The function assumes that POSE_CONNECTIONS, LEFT_HAND_CONNECTIONS, and RIGHT_HAND_CONNECTIONS are predefined lists
      of landmark index pairs that define the connections for pose and hands.
    """
    # Extract pose landmarks and compute pose distances
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_landmarks]).flatten()
        pose_distances = compute_distances(pose_landmarks, POSE_CONNECTIONS)
    else:
        pose = np.zeros(33 * 4)
        pose_distances = np.zeros(len(POSE_CONNECTIONS))

    # Extract left hand landmarks and compute left hand distances
    if results.left_hand_landmarks:
        left_hand_landmarks = results.left_hand_landmarks.landmark
        lh = np.array([[res.x, res.y, res.z] for res in left_hand_landmarks]).flatten()
        lh_distances = compute_distances(left_hand_landmarks, LEFT_HAND_CONNECTIONS)
    else:
        lh = np.zeros(21 * 3)
        lh_distances = np.zeros(len(LEFT_HAND_CONNECTIONS))

    # Extract right hand landmarks and compute right hand distances
    if results.right_hand_landmarks:
        right_hand_landmarks = results.right_hand_landmarks.landmark
        rh = np.array([[res.x, res.y, res.z] for res in right_hand_landmarks]).flatten()
        rh_distances = compute_distances(right_hand_landmarks, RIGHT_HAND_CONNECTIONS)
    else:
        rh = np.zeros(21 * 3)
        rh_distances = np.zeros(len(RIGHT_HAND_CONNECTIONS))

    # Initialize list for proximity features
    proximity_features = []

    # Compute distance between left wrist and nose if both landmarks are available
    if results.pose_landmarks and results.left_hand_landmarks:
        left_wrist = results.left_hand_landmarks.landmark[0]  # Assuming index 0 is the wrist
        nose = results.pose_landmarks.landmark[0]  # Assuming index 0 is the nose
        distance = np.linalg.norm(
            np.array([left_wrist.x, left_wrist.y, left_wrist.z]) -
            np.array([nose.x, nose.y, nose.z])
        )
        proximity_features.append(distance)
    else:
        proximity_features.append(0.0)

    # Compute distance between right wrist and nose if both landmarks are available
    if results.pose_landmarks and results.right_hand_landmarks:
        right_wrist = results.right_hand_landmarks.landmark[0]  # Assuming index 0 is the wrist
        nose = results.pose_landmarks.landmark[0]  # Assuming index 0 is the nose
        distance = np.linalg.norm(
            np.array([right_wrist.x, right_wrist.y, right_wrist.z]) -
            np.array([nose.x, nose.y, nose.z])
        )
        proximity_features.append(distance)
    else:
        proximity_features.append(0.0)

    # Concatenate all features into a single feature vector
    keypoints = np.concatenate([pose, lh, rh])
    connection_features = np.concatenate([pose_distances, lh_distances, rh_distances])
    proximity_features = np.array(proximity_features)

    # Final feature vector
    feature_vector = np.concatenate([keypoints, connection_features, proximity_features])

    return feature_vector



def process_hand_landmarks(results, hand_analyzer):
    """
    Processes hand landmarks from MediaPipe results, analyzing palm orientation, hand rotation,
    and hand shape for both left and right hands if present. Concatenates the extracted features
    into a single keypoints array.

    Parameters:
    - results: MediaPipe results object with hand landmarks.
    - hand_analyzer: An object with methods to calculate palm normal, classify hand view,
                     calculate hand rotation, and determine hand shape.

    Returns:
    - keypoints: A numpy array containing the concatenated keypoints and encoded hand analysis results.
    - sequence: A list to which the final keypoints are appended.
    """

    # Initialize sequence list and keypoints extraction from results
    keypoints = extract_keypoints(results)

    left_hand_orientation_encoded = [0, 0, 0, 0, 0]
    right_hand_orientation_encoded = [0, 0, 0, 0, 0]
    calculate_left_rotation_encoded = [0] * 8
    calculate_right_rotation_encoded = [0] * 8
    determine_lhand_shape_encoded = [0, 0, 0, 0]
    determine_rhand_shape_encoded = [0, 0, 0, 0]

    # Process left hand landmarks if available
    if results.left_hand_landmarks:
        normal_vector_left = hand_analyzer.calculate_palm_normal(results.left_hand_landmarks, 'Left')
        left_hand_orientation_encoded = hand_analyzer.classify_hand_view(normal_vector_left, 'Left')
        calculate_left_rotation_encoded, bucket = hand_analyzer.calculate_hand_rotation(results.left_hand_landmarks)
        determine_lhand_shape_encoded = hand_analyzer.determine_hand_shape(results.left_hand_landmarks)

    # Process right hand landmarks if available
    if results.right_hand_landmarks:
        normal_vector_right = hand_analyzer.calculate_palm_normal(results.right_hand_landmarks, 'Right')
        right_hand_orientation_encoded = hand_analyzer.classify_hand_view(normal_vector_right, 'Right')
        calculate_right_rotation_encoded, bucket = hand_analyzer.calculate_hand_rotation(results.right_hand_landmarks)
        determine_rhand_shape_encoded = hand_analyzer.determine_hand_shape(results.right_hand_landmarks)

    # Concatenate all extracted features
    keypoints = np.concatenate([
        keypoints,
        left_hand_orientation_encoded, right_hand_orientation_encoded,
        calculate_left_rotation_encoded, calculate_right_rotation_encoded,
        determine_lhand_shape_encoded, determine_rhand_shape_encoded
    ])


    return keypoints
