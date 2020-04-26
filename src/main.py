from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtMultimedia import *
from PySide2.QtMultimediaWidgets import *
import sys
import os
import multiprocessing

from video_controls import *
from recognizer import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cel Recognizer")

        self.worker = CelRecognizerWorker()
        self.worker.update.connect(self.recognizer_results)

        self.cel_path = None
        self.video_path = None

        # Create seek shortcuts
        QShortcut(QKeySequence(Qt.Key_Left), self, self.seek_left)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.seek_right)

        # Create open cel action
        openCelAction = QAction(QIcon('open.png'), '&Open Cel', self)
        openCelAction.setShortcut('Ctrl+O')
        openCelAction.setStatusTip('Open movie')
        openCelAction.triggered.connect(self.openCel)

        # Create open movie action
        openMovieAction = QAction(QIcon('open.png'), 'Open &Movie', self)
        openMovieAction.setShortcut('Ctrl+M')
        openMovieAction.setStatusTip('Open movie')
        openMovieAction.triggered.connect(self.openMovie)

        # Create exit action
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exit)

        # Create menu bar and add action
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        fileMenu.addAction(openCelAction)
        fileMenu.addAction(openMovieAction)
        fileMenu.addAction(exitAction)

        # Create video player & controls
        videoWidget = QVideoWidget()

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setVideoOutput(videoWidget)

        self.videoControls = VideoControls(self.mediaPlayer)

        # Create layout for the video
        videoLayout = QVBoxLayout()
        videoLayout.addWidget(videoWidget, stretch=1)
        videoLayout.addWidget(self.videoControls, stretch=0)

        wid = QWidget()
        wid.setLayout(videoLayout)
        self.setCentralWidget(wid)

    def openCel(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Cel", QDir.homePath())
        
        if fileName != '':
            self.cel_path = fileName
            self.restart_recognizer()

    def openMovie(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Movie", QDir.homePath())

        if fileName != '':
            self.video_path = fileName
            self.mediaPlayer.setMedia(
                QMediaContent(QUrl.fromLocalFile(fileName)))
            self.restart_recognizer()

    def exit(self):
        self.close()
    
    def restart_recognizer(self):
        print(self.cel_path, self.video_path)
        if self.cel_path is not None and self.video_path is not None:
            self.worker.submit(self.cel_path, self.video_path)

    def recognizer_results(self, results: [(float, float)]):
        self.videoControls.recognizer_results(results)
    
    def seek_left(self):
        self.videoControls.seek_frames(-1)

    def seek_right(self):
        self.videoControls.seek_frames(1)


def cleanup():
    for proc in multiprocessing.active_children():
        proc.terminate()
        proc.join()
        proc.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'

    app = QApplication(sys.argv)
    
    player = MainWindow()
    player.resize(640, 480)
    player.show()

    exit_code = app.exec_()
    cleanup()
    sys.exit(exit_code)
