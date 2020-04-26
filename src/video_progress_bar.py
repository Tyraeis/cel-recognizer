from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtGui import *
import numpy as np
import math
from recognizer import CelRecognizer

class VideoProgressBar(QWidget):
    seek = Signal(float)

    progress : float
    length : float
    recognizer_results : [(float, float)]
    batch_duration = 1000

    def __init__(self):
        super().__init__()
        self.progress = 0
        self.length = 0
        self.recognizer_results = []
    
    def setResults(self, results: [(float, float)]):
        self.recognizer_results = sorted(results, key=lambda x: -x[1])
        self.update()
    
    def setProgress(self, progress: float):
        self.progress = progress
        self.update()
    
    def setLength(self, length: float):
        self.length = length
        self.update()

    def minimumSizeHint(self):
        return QSize(8, 8)

    def mousePressEvent(self, evt: QMouseEvent):
        if evt.button() == Qt.LeftButton:
            new_progress = (evt.x() / self.width()) * self.length
            self.seek.emit(new_progress)
    
    def mouseMoveEvent(self, evt: QMouseEvent):
        if evt.buttons() & Qt.LeftButton > 0:
            new_progress = (evt.x() / self.width()) * self.length
            self.seek.emit(new_progress)
    
    def paintEvent(self, evt):
        painter = QPainter()
        painter.begin(self)
        self.drawBar(painter)
        painter.end()

    def drawBar(self, painter: QPainter):
        width = self.width()
        height = self.height()

        # Fill in background with black
        painter.fillRect(0, 0, width, height, QColor.fromRgb(0, 0, 0))

        if self.length == 0:
            return

        if len(self.recognizer_results) > 0:
            batch_width = math.ceil((self.batch_duration / self.length) * width)
            # Calculate mean/standard deviation of the results
            mean = np.mean(self.recognizer_results, axis=0)[1]
            std = np.std(self.recognizer_results, axis=0)[1]
            for time, dist in self.recognizer_results:
                # Calculate the position of the rectangle
                x = (time / self.length) * width

                # Choose a color based on the # of standard deviations this distance
                # is away from the mean
                confidence = max(0, (mean - dist) / std)
                if confidence > 3:
                    color = QColor.fromRgb(255, 0, 0)
                elif confidence > 2:
                    color = QColor.fromRgb(255, 128, 0)
                elif confidence > 1:
                    color = QColor.fromRgb(255, 255, 0)
                else:
                    color = QColor.fromRgb(0, 255, 0)

                # Draw this rectangle
                painter.fillRect(x, 0, batch_width, height, color)

        # Draw a marker for the current position
        positionX = (self.progress / self.length) * width
        painter.fillRect(positionX, 0, 2, height, QColor.fromRgb(255, 255, 255))
