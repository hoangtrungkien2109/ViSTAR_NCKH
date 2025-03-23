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
    
    if np.linalg.norm(prev_frame - post_frame) <= 1:
        middle = np.linspace(prev_frame, post_frame, num=3)
    elif np.linalg.norm(prev_frame - post_frame) <= 2:
        middle = np.linspace(prev_frame, post_frame, num=5)
    else:
        middle = np.linspace(prev_frame, post_frame, num=15)
    
    concatenated_frame = np.concatenate([middle, post_frame],axis=0)
    return concatenated_frame
  
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
                result = concatenate_frame(prev_frame=prev_frame, post_frame=post_frame)
                
                result = result.tolist()
                
                self.processed_frame_queue.extend(result)
                
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