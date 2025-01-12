import cv2
import numpy as np
import mediapipe as mp
import math
from typing import List, Tuple, Union


class HandAnalyzer:
    """
    A class for analyzing hand landmarks and extracting various hand features.

    Parameters
    ----------
    label_encoded : bool, optional
        If True, outputs are label-encoded as one-hot vectors.
        If False, outputs are string labels. Default is True.
    """

    def __init__(self, label_encoded: bool = True):
        self.label_encoded = label_encoded
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.left_hand_orientation_encoded = [0, 0, 0, 0, 0]
        self.right_hand_orientation_encoded = [0, 0, 0, 0, 0]
        self.calculate_left_rotation_encoded = [0] * 8
        self.calculate_right_rotation_encoded = [0] * 8
        self.determine_lhand_shape_encoded = [0, 0, 0, 0]
        self.determine_rhand_shape_encoded = [0, 0, 0, 0]

    def calculate_palm_normal(
        self,
        hand_landmarks,
        handedness_label: str,
    ) -> np.ndarray:
        """
        Calculate the normal vector of the palm based on hand landmarks.

        Parameters
        ----------
        hand_landmarks : NormalizedLandmarkList
            Landmarks of the hand detected by MediaPipe.
        handedness_label : str
            Label indicating 'Left' or 'Right' hand.

        Returns
        -------
        np.ndarray
            Normalized normal vector of the palm.
        """
        index_mcp = np.array(
            [
                hand_landmarks.landmark[5].x,
                hand_landmarks.landmark[5].y,
                hand_landmarks.landmark[5].z,
            ]
        )
        wrist = np.array(
            [
                hand_landmarks.landmark[0].x,
                hand_landmarks.landmark[0].y,
                hand_landmarks.landmark[0].z,
            ]
        )
        pinky_mcp = np.array(
            [
                hand_landmarks.landmark[17].x,
                hand_landmarks.landmark[17].y,
                hand_landmarks.landmark[17].z,
            ]
        )

        if handedness_label == "Right":
            vector1 = index_mcp - wrist
            vector2 = pinky_mcp - wrist
        else:
            vector1 = pinky_mcp - wrist
            vector2 = index_mcp - wrist

        normal_vector = np.cross(vector1, vector2)
        normal_vector /= np.linalg.norm(normal_vector)
        return normal_vector

    def classify_hand_view(
        self, normal_vector: np.ndarray, handedness_label: str
    ) -> Union[List[int], str]:
        """
        Classify the orientation of the hand based on the palm normal vector.

        Parameters
        ----------
        normal_vector : np.ndarray
            Normalized normal vector of the palm.
        handedness_label : str
            Label indicating 'Left' or 'Right' hand.

        Returns
        -------
        Union[List[int], str]
            One-hot encoded list or string label representing the hand orientation.
        """
        nx, ny, nz = normal_vector
        threshold = 0.7
        side_threshold = 0.3

        if nz > threshold:
            # Palm view
            return [1, 0, 0, 0, 0] if self.label_encoded else "Palm"
        elif nz < -threshold:
            # Back view
            return [0, 1, 0, 0, 0] if self.label_encoded else "Back"
        else:
            if handedness_label == "Right":
                if nx > side_threshold:
                    # Sideways-left (right hand viewed from the side)
                    return [0, 0, 1, 0, 0] if self.label_encoded else "Sideways-left"
                elif nx < -side_threshold:
                    # Sideways-right
                    return [0, 0, 0, 1, 0] if self.label_encoded else "Sideways-right"
            else:
                if nx > side_threshold:
                    # Sideways-right
                    return [0, 0, 0, 1, 0] if self.label_encoded else "Sideways-right"
                elif nx < -side_threshold:
                    # Sideways-left
                    return [0, 0, 1, 0, 0] if self.label_encoded else "Sideways-left"
            # Undefined orientation
            return [0, 0, 0, 0, 1] if self.label_encoded else "Other-Sideways"

    def calculate_hand_rotation(
        self, hand_landmarks
    ) -> Union[Tuple[List[int], int], Tuple[float, int]]:
        """
        Calculate the rotation angle of the hand based on landmarks.

        Parameters
        ----------
        hand_landmarks : NormalizedLandmarkList
            Landmarks of the hand detected by MediaPipe.

        Returns
        -------
        Union[Tuple[List[int], int], Tuple[float, int]]
            One-hot encoded list and bucket index or angle in degrees and bucket index.
        """
        mmcp = np.array(
            [
                hand_landmarks.landmark[9].x,
                hand_landmarks.landmark[9].y,
            ]
        )
        wrist = np.array(
            [
                hand_landmarks.landmark[0].x,
                hand_landmarks.landmark[0].y,
            ]
        )

        vector = mmcp - wrist
        angle = math.degrees(math.atan2(vector[1], vector[0]))
        angle = (angle + 360) % 360

        bucket = int(((angle + 22.5) % 360) / 45) % 8
        one_hot_bucket = [0] * 8
        one_hot_bucket[bucket] = 1

        if self.label_encoded:
            return one_hot_bucket, bucket
        else:
            return angle, bucket

    def determine_hand_shape(
        self, hand_landmarks
    ) -> Union[List[int], str]:
        """
        Determine the hand shape based on the extension of fingers.

        Parameters
        ----------
        hand_landmarks : NormalizedLandmarkList
            Landmarks of the hand detected by MediaPipe.

        Returns
        -------
        Union[List[int], str]
            One-hot encoded list or string label representing the hand shape.
        """

        def is_finger_extended(mcp, pip, dip, tip) -> bool:
            return tip.y < pip.y < mcp.y

        def is_thumb_extended(
            thumb_cmc, thumb_mcp, thumb_ip, thumb_tip
        ) -> bool:
            thumb_tip_coords = np.array([thumb_tip.x, thumb_tip.y])
            thumb_ip_coords = np.array([thumb_ip.x, thumb_ip.y])
            thumb_mcp_coords = np.array([thumb_mcp.x, thumb_mcp.y])

            thumb_dir = thumb_tip_coords - thumb_mcp_coords
            thumb_ip_dir = thumb_ip_coords - thumb_mcp_coords

            dot_product = np.dot(thumb_dir, thumb_ip_dir)
            magnitude = np.linalg.norm(thumb_dir) * np.linalg.norm(thumb_ip_dir)
            if magnitude == 0:
                return False
            angle = np.arccos(dot_product / magnitude)
            return angle < np.deg2rad(25)

        # Get landmarks for the thumb
        thumb_cmc = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

        # Get landmarks for the index finger
        index_mcp = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP
        ]
        index_pip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP
        ]
        index_dip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.INDEX_FINGER_DIP
        ]
        index_tip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP
        ]

        # Get landmarks for the middle finger
        middle_mcp = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP
        ]
        middle_pip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP
        ]
        middle_dip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP
        ]
        middle_tip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
        ]

        # Get landmarks for the ring finger
        ring_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        ring_dip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]

        # Get landmarks for the pinky finger
        pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        pinky_dip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        # Determine if fingers are extended
        thumb_extended = is_thumb_extended(
            thumb_cmc, thumb_mcp, thumb_ip, thumb_tip
        )
        index_extended = is_finger_extended(
            index_mcp, index_pip, index_dip, index_tip
        )
        middle_extended = is_finger_extended(
            middle_mcp, middle_pip, middle_dip, middle_tip
        )
        ring_extended = is_finger_extended(ring_mcp, ring_pip, ring_dip, ring_tip)
        pinky_extended = is_finger_extended(
            pinky_mcp, pinky_pip, pinky_dip, pinky_tip
        )

        extended_fingers = [
            thumb_extended,
            index_extended,
            middle_extended,
            ring_extended,
            pinky_extended,
        ]

        if all(extended_fingers):
            # Open palm
            return [1, 0, 0, 0] if self.label_encoded else "Open palm"
        elif not any(extended_fingers):
            # Closed fist
            return [0, 1, 0, 0] if self.label_encoded else "Fist"
        elif index_extended and not any(
            [middle_extended, ring_extended, pinky_extended]
        ):
            # Pointing index
            return [0, 0, 1, 0] if self.label_encoded else "Pointing"
        else:
            # Other hand shape
            return [0, 0, 0, 1] if self.label_encoded else "Other hand shape"
