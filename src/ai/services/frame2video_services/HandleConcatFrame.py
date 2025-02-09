from collections import deque
import numpy as np
from loguru import logger

def concatenate_frame(prev_frame, post_frame):
    """
    Args:
    prev_frame: only frame (75,3) which is that last frame of a previous word
    post_frame: only frame (75,3) which is frist frame of post word
    """
    prev_frame = np.array(prev_frame)
    post_frame = np.array(post_frame)
    
    # prev_frame = prev_frame[~np.any(prev_frame == 0, axis=(1,2))]
    # post_frame = post_frame[~np.any(post_frame == 0, axis=(1,2))]

    if np.linalg.norm(prev_frame - post_frame) <= 1:
        middle = np.linspace(prev_frame[-1], post_frame, num=7)
    elif np.linalg.norm(prev_frame - post_frame) <= 2:
        middle = np.linspace(prev_frame, post_frame, num=10)
    else:
        middle = np.linspace(prev_frame, post_frame, num=15)
    
    return middle

    # concatenated_landmarks = np.concatenate(all_landmarks,axis=0)
    # return concatenated_landmarks
  
class HandleConcatFrame:
    def __init__(self):
        """
        INIT
        """
        self.processed_frame_queue = deque()
        
    def push_into_process_queue(self, frames):
        """
        push frame into queue with process progress
        """
        try:
            if (self.processed_frame_queue[-1] is not None):
                logger.info("Start concat")
                prev_frame = self.processed_frame_queue.pop()
                post_frame = frames[0]
                
                # Concat 2 frame
                middle = concatenate_frame(prev_frame=prev_frame, post_frame=post_frame)
                logger.info(f"Middle: {middle.shape}")
                logger.info(f"post_frame: {post_frame.shape}")
                
                concatenated_frame = np.concatenate([middle, frames],axis=0)
                concatenated_frame = concatenated_frame.tolist()
                
                # Push into queue
                self.processed_frame_queue.extend(concatenated_frame)
                
                logger.success("Processed successfuly")
        except IndexError:
            logger.error("Some error")
            self.processed_frame_queue.extend(frames)
    
    def pop(self):
        try:
            return self.processed_frame_queue.pop()
        except IndexError:
            return None