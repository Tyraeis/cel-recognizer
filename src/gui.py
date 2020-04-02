import sys
import PySide2
from PySide2.QtCore import QUrl
from PySide2.QtWidgets import QApplication
from PySide2.QtMultimedia import QMediaPlayer
from PySide2.QtMultimediaWidgets import QVideoWidget

if __name__ == '__main__':
    DATA_DIR = 'C:/Users/Noah/Documents/se_data'
    VIDEO_PATH = DATA_DIR + '/videos/Thundercats/sword_in_a_hole.avi'

    app = QApplication(sys.argv)
    video = QVideoWidget()

    player = QMediaPlayer()
    player.setMedia(QUrl.fromLocalFile(VIDEO_PATH))
    player.setVideoOutput(video)


    def video_available_changed(available):
        if available:
            video.updateGeometry()
            video.adjustSize()


    player.mediaStatusChanged.connect(video_available_changed)

    video.show()
    player.play()
    app.exec_()
    player.stop()
