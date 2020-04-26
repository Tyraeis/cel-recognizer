import os
import json
from video import Video

class VideoLibrary:
    def __init__(self, path):
        self._path = path
    
    def list_videos(self):
        return [
            p[:-4]
            for p in os.listdir(self._path)
            if p[-4:] == '.avi'
        ]
    
    def get_video(self, name):
        return Video(os.path.join(self._path, name) + '.avi')
    
    def save_checkpoint(self, data):
        with open(os.path.join(self._path, '.checkpoint.json'), 'w') as f:
            json.dump(data, f)
    
    def load_checkpoint(self):
        with open(os.path.join(self._path, '.checkpoint.json'), 'r') as f:
            return json.load(f)