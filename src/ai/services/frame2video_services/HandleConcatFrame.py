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
    
    logger.info(f"Prev:{prev_frame.shape}")
    logger.info(f"Post:{post_frame.shape}")
    
    if np.array_equal(prev_frame, post_frame):
        logger.info("2 frame equal")
        middle = np.linspace(prev_frame, post_frame, num=2)
        return middle
    
    if np.linalg.norm(prev_frame - post_frame) <= 1:
        middle = np.linspace(prev_frame, post_frame, num=3)
    elif np.linalg.norm(prev_frame - post_frame) <= 2:
        middle = np.linspace(prev_frame, post_frame, num=5)
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
                prev_frame = self.processed_frame_queue.pop()
                post_frame = frames[0]
                
                # Concat 2 frame
                middle = concatenate_frame(prev_frame=prev_frame, post_frame=post_frame)
                
                concatenated_frame = np.concatenate([middle, frames],axis=0)
                concatenated_frame = concatenated_frame.tolist()
                
                self.processed_frame_queue.extend(concatenated_frame)
                
                logger.success("Processed successfuly")
        except IndexError:
            self.processed_frame_queue.extend(frames)
        except Exception as e:
            print(e)
    
    def pop(self):
        try:
            a = np.array([self.processed_frame_queue.pop()])
            logger.info(f"len: {len(self.processed_frame_queue)}")
            return a
        except IndexError:
            return None