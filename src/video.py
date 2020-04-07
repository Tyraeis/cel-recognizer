import cv2

class Video:
    def __init__(self, path):
        self._path = path
        self._capture = cv2.VideoCapture(path)
    
    def get_fps(self):
        """Returns the FPS of the video"""
        return self._capture.get(cv2.CAP_PROP_FPS)
    
    def get_size(self):
        """Returns the size of the video as a tuple of (width, height)"""
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    
    def get_frame_count(self):
        """Returns the total number of frames in the video"""
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_current_frame(self):
        """Retrieves a frame from the video and returns it as a numpy array.
        This should be called while using the frames() iterator."""
        ret, frame = self._capture.retrieve()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            return None
    
    def get_current_frame_num(self):
        """Returns the index of the current frame.
        This should be called while using the frames() iterator."""
        return int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
    
    def get_current_time(self):
        """Returns the current time in milliseconds.
        This should be called while using the frames() iterator."""
        return self._capture.get(cv2.CAP_PROP_POS_MSEC)

    def frames(self):
        """An iterator over the frames in this video. The iterator yields the frame index, you
        should use get_current_frame() to get the actual frame."""
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret = self._capture.grab()
            if ret:
                yield self.get_current_frame_num()
            else:
                break
