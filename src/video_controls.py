

from PySide2.QtCore import *
from PySide2.QtMultimedia import *
from PySide2.QtMultimediaWidgets import *
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from os import path
import sys

from video_progress_bar import *

def format_time(ms):
    s = (ms // (1000)) % 60
    m = (ms // (60*1000)) % 60
    h = (ms // (60*60*1000))

    if h > 0:
        return '{}:{:02d}:{:02d}'.format(h, m, s)
    else:
        return '{}:{:02d}'.format(m, s)

class VideoControls(QWidget):
    error = Signal()

    def __init__(self, mediaPlayer : QMediaPlayer, parent : QWidget=None):
        super().__init__(parent)
        self.setWindowTitle("Cel Recognizer") 

        self.mediaPlayer = mediaPlayer

        # Play button that plays/pauses the video
        self.playButton = QPushButton()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)
        self.playButton.setFixedSize(32,32)

        # Label displaying the current time / video length
        self.timeLabel = QLabel()
        self.timeLabel.setText("0:00 / 0:00")

        # Progress bar to show where the video is currently
        self.progressBar = VideoProgressBar()
        self.progressBar.seek.connect(self.setPosition)

        # Create a layout to hold the time label / slider
        sliderLayout = QHBoxLayout()
        sliderLayout.addWidget(self.timeLabel, stretch=0)
        sliderLayout.addWidget(self.progressBar, stretch=1)

        # Create a layout to hold the button centered under video
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.playButton)  

        # Create a layout for the progress bar and controls
        videoLayout = QVBoxLayout()
        videoLayout.addLayout(sliderLayout)
        videoLayout.addLayout(buttonLayout)

        # Set widget to contain window contents
        self.setLayout(videoLayout)

        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

    def _get_framerate(self):
        if 'VideoFrameRate' in self.mediaPlayer.availableMetaData():
            return self.mediaPlayer.metaData('VideoFrameRate')
        else:
            return 1

    def seek_frames(self, delta_frames):
        fps = self._get_framerate()
        ms_per_frame = 1000 / fps
        current_position = self.mediaPlayer.position()
        self.setPosition(current_position + delta_frames * ms_per_frame)

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        positionText = format_time(position)
        durationText = format_time(self.mediaPlayer.duration())
        self.timeLabel.setText('{} / {}'.format(positionText, durationText))
        self.progressBar.setProgress(position)

    def durationChanged(self, duration):
        self.progressBar.setLength(duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        print(self.mediaPlayer.errorString())
        self.error.emit(self.mediaPlayer.errorString())
    
    def recognizer_results(self, results: [(float, float)]):
        self.progressBar.setResults(results)
