import os
import os.path
from os.path import join
import random

import cv2
import numpy as np
import skimage
from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_ubyte
from skimage.transform import resize
from skimage.color import rgb2ycbcr
from skimage.exposure import adjust_gamma, adjust_log

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, apply_affine_transform

BATCH_SIZE = 128

def resize_maintain_ar(image, target_size):
    """Resizes a given image to target_size, maintaining its aspect ratio by cropping its edges"""
    target_width, target_height = target_size
    current_height, current_width, _ = image.shape
    
    current_ar = current_width / current_height
    target_ar = target_width / target_height
    if current_ar > target_ar:
        half_resized_width = int(current_height * target_ar) // 2
        half_width = current_width // 2
        
        image = image[:, half_width - half_resized_width:half_width + half_resized_width, :]
    elif current_ar < target_ar:
        half_resized_height = int(current_width / target_ar) // 2
        half_height = current_height // 2
        
        image = image[half_height - half_resized_height:half_height + half_resized_height, :]
    
    return resize(image, [target_height, target_width])

class CelRecognizer:
    def __init__(self, work_dir, cel, frame_size):
        """CelRecognizer constructor

        Arguments:
            work_dir: Path to a temporary directory used to store intermediate files
            cel: A numpy array containing the cel image
            frame_size: A tuple of `(width, height)`. This is used as the size of the input to the model
                This size may be different than the actual size of the frames given to the CelRecognizer,
                all frames will be resized to this size.
        """
        self.work_dir = work_dir
        self.frame_size = frame_size
        self.cel = resize_maintain_ar(img_as_float(cel), frame_size)
        self.frame_num = 0

        self._build_model()
        self._generate_mask()
        self._make_dirs()

    
    def _make_dirs(self):
        """Creates required directories inside work_dir"""
        os.makedirs(join(self.work_dir, 'with_cel'), exist_ok=True)
        os.makedirs(join(self.work_dir, 'no_cel'), exist_ok=True)
    
    def _generate_mask(self):
        """Generates a background mask for the cel by chroma-keying it"""
        # TODO: generate chroma_key_range based on user-provided background samples
        # This chroma_key_range was based on the Sword In A Hole cel
        chroma_key_range = [
            np.array([193.4, 123.6, 124.0]),
            np.array([255.0, 133.2, 132.3])
        ]
        cel_ycbcr = rgb2ycbcr(self.cel)
        self.mask = img_as_float(cv2.inRange(cel_ycbcr, chroma_key_range[0], chroma_key_range[1]))
        self.mask = np.expand_dims(self.mask, 2)

    def _build_model(self):
        """Initializes the machine learning model"""
        self.model = Sequential([
            Conv2D(16, 3, padding='same', activation='relu', input_shape=(self.frame_size[1], self.frame_size[0], 3)),
            MaxPooling2D(),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D(),
            Dropout(0.2),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def _predict(self, frame_iterator):
        """Get predictions for a sequence of frames
        
        Arguments:
            frame_iterator: an iterator over the frames that should be sent to the model.
                The frames should be numpy arrays in RGB format and an arbitrary datatype
            
        Returns an iterator over batches of predictions. Each batch is an array containing an arbitrary
        number of predictions, one prediction per frame yielded by frame_iterator.
        """
        batch = np.ndarray((BATCH_SIZE, self.frame_size[1], self.frame_size[0], 3))
        batch_index = 0
        for frame in frame_iterator:
            frame = cv2.resize(frame, self.frame_size).astype('float')
            batch[batch_index] = frame
            batch_index += 1

            if batch_index == BATCH_SIZE:
                yield self.model.predict(batch)
                batch_index = 0
        
        yield self.model.predict(batch[:batch_index])
    
    def _get_transformed_cel(self):
        """Creates an augmented cel and mask for use in data generation"""
        # Color transformation
        gamma = random.uniform(0.2, 1.8)
        cel_t = adjust_gamma(self.cel, gamma=gamma)
        gain = random.uniform(0.2, 1.0)
        cel_t = adjust_log(cel_t, gain=gain)

        # Spatial transformation
        theta = random.gauss(0, 5)
        zoom = random.uniform(0.8, 1.0)
        cel_t = apply_affine_transform(cel_t, theta=theta, zx=zoom, zy=zoom, fill_mode='constant')
        mask_t = apply_affine_transform(self.mask, theta=theta, zx=zoom, zy=zoom, fill_mode='constant')
    
        return cel_t, mask_t
    
    def generate_training_data(self, frame_iterator):
        """Generates frames with and without the cel for use in training
        
        Arguments:
            frame_iterator: an iterator over the frames that should be sent to the model.
                The frames should be numpy arrays in RGB format and an arbitrary datatype
        """
        for frame in frame_iterator:
            cel_t, mask_t = self._get_transformed_cel()
            frame = img_as_float(resize(frame, self.frame_size[::-1]))
            modified_frame = (cel_t * (1 - mask_t)) + (frame * mask_t)

            imsave(join(self.work_dir, 'no_cel', f'{self.frame_num:06d}.jpg'), img_as_ubyte(frame))
            imsave(join(self.work_dir, 'with_cel', f'{self.frame_num:06d}.jpg'), img_as_ubyte(modified_frame))

            self.frame_num += 1

        print(f'Generated {2*self.frame_num} training frames')
    
    def train(self, epochs):
        """Trains the model using previously generated training data

        Arguments:
            epochs: the number of epochs to train
        """
        image_generator = ImageDataGenerator()
        data_gen = image_generator.flow_from_directory(
            batch_size=BATCH_SIZE,
            directory=self.work_dir,
            shuffle=True,
            target_size=self.frame_size[::-1],
            class_mode='binary',
            classes=['no_cel', 'with_cel']
        )

        history = self.model.fit(
            data_gen,
            epochs=epochs
        )

        return history
    
    def find_possible_matches(self, frame_iterator, merge_threshold=30):
        last_match = None
        frame_index = 0
        # Iterate over the predictions from the model
        for batch in self._predict(frame_iterator):
            for confidence in batch:
                # If the model thinks the cel is in this frame:
                if confidence > 0.5:
                    # Merge this into the last match if it wasn't too long ago, otherwise yield the match
                    if last_match is None:
                        # This is the first match found
                        last_match = (frame_index, frame_index)
                    elif frame_index - last_match[1] < merge_threshold:
                        # This can be merged into the last match
                        last_match = (last_match[0], frame_index)
                    else:
                        # This is a new match; yield the last match and make a new one
                        yield last_match
                        last_match = (frame_index, frame_index)
                    
                frame_index += 1
        # Yield the last match if there was one
        if last_match is not None:
            yield last_match


def frames(video_file):
    video = cv2.VideoCapture(video_file)
    while True:
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
        else:
            break
    video.release()

def every_nth(iterator, n):
    for i, value in enumerate(iterator):
        if i % n == 0:
            yield value


if __name__ == '__main__':
    data_dir = 'C:/Users/Noah/Documents/se_data'
    work_dir = data_dir + '/work'
    cel_path = data_dir + '/cels/Lion-O - Sword In A Hole/Lion-O.jpg'
    video_path = data_dir + '/videos/Thundercats/Thundercats_S1_Ep35_–_Sword_in_a_Hole.mp4'

    import math
    fps = 23.98
    def format_time(frame_num):
        s = frame_num / fps
        m = s // 60
        s = math.floor(s % 60)
        return f'{m:d}:{s:02d}'

    cel = imread(cel_path)
    cr = CelRecognizer(work_dir, cel, (120, 90))
    cr.generate_training_data(every_nth(frames(video_path), 150))
    cr.train(10)

    for start, end in cr.find_possible_matches(frames(video_path)):
        print(f'{format_time(start)} - {format_time(end)}')
