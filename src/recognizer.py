import math
import numpy as np
import cv2
from video import Video

class CelRecognizer:
    def __init__(self, cel):
        """CelRecognizer contstructor

        Arguments:
            cel: A numpy array containing the cel image
        """
        self._original_cel = cel
        self._orb = cv2.ORB_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def _set_size(self, expected_size):
        """Resizes the cel to approximately match a given size"""
        cel_ar = self._original_cel.shape[1] // self._original_cel.shape[0]
        self._cel = cv2.resize(self._original_cel, (expected_size[0], expected_size[1] * cel_ar))
        self._cel_kp, self._cel_desc = self._orb.detectAndCompute(self._cel, None)
    
    def _match_features(self, image):
        """Finds matching features between the cel and a given image.

        Returns the array of match objects and the array of keypoints
        """
        kp, desc = self._orb.detectAndCompute(image, None)
        if desc is None:
            return None, None
        matches = self._matcher.match(self._cel_desc, desc)
        return matches, kp
    
    def _distance(self, image):
        """Computes the distance between the cel and a given image"""
        matches, kp = self._match_features(image)
        
        if matches is None or len(matches) < 10:
            return math.inf
        
        return sum(x.distance for x in matches) / len(matches)
    
    def _find_outliers(self, distances):
        """Finds outliers in a list of batches.
        Outliers are considered to be points where the value is more than 4 standard deviations below the mean.
        
        Arguments:
            results: An array of tuples, where the first item is an arbitrary label and the second item is the
                value to find outliers over.
        """
        # Mean and standard deviation only make sense on non-empty arrays
        if len(distances) == 0:
            return []
        # Get the mean and standard deviation of the distances
        mean = np.mean(distances, axis=0)[1]
        std = np.std(distances, axis=0)[1]
        # Note: the lack of an absolute value in finding the deviance from the mean is intentional.
        # We only care about outliers that are below the mean, as those are the ones that could be matches.
        return [ (label, dist) for label, dist in distances if mean - dist > 4 * std ]
    
    def find_possible_matches(self, video, batch_duration=1000, on_batch=None):
        """Finds possible uses of the cel in a video.

        Arguments:
            video: A Video instance to search for matches in
            batch_duration: The length in milliseconds for each batch
            on_batch: A callback that is called after every batch. Two arguments are passed to this
            function:
                batch: the minimum distance within the last batch
                matches: An array of batches that might be matches.

        Returns an array of batches that might be matches.
        """
        self._set_size(video.get_size())

        distances = []

        batch_index = 0
        batch_min_dist = math.inf
        for frame_num in video.frames():
            frame = video.get_current_frame()
            current_time = video.get_current_time()

            # Check if a new batch needs to be started
            if current_time // batch_duration != batch_index:
                # Add the batch to the results (unless all of the distances were infinite)
                if batch_min_dist != math.inf:
                    distances.append((batch_index * batch_duration, batch_min_dist))

                # Run the on_batch callback if given
                if on_batch is not None:
                    on_batch(batch_min_dist, self._find_outliers(distances))
                
                # Start a new batch
                batch_index = current_time // batch_duration
                batch_min_dist = self._distance(frame)
            else:
                # Add the distance for this frame to the batch
                batch_min_dist = min(batch_min_dist, self._distance(frame))
        
        # Add the last batch to the results
        if batch_min_dist != math.inf:
            distances.append((batch_index * batch_duration, batch_min_dist))
        
        # Run the on_batch callback if given
        if on_batch is not None:
            on_batch(batch_min_dist, self._find_outliers(distances))

        # Return all of the batches that were outliers below the mean, as those are the
        # ones that might be a match
        return self._find_outliers(distances)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Find uses of an animation cel in one or more videos')
    parser.add_argument('cel', help='path to an image of the cel')
    parser.add_argument('video', nargs='+', help='paths to video files to search')
    args = parser.parse_args()

    def format_time(fps, frame_num):
        s = frame_num / fps
        m = s // 60
        s = math.floor(s % 60)
        return f'{int(m)}:{s:02d}'

    cel = cv2.cvtColor(cv2.imread(args.cel), cv2.COLOR_BGR2GRAY)
    recognizer = CelRecognizer(cel)

    def print_progress(video, outliers, bar_length=40):
        fps = video.get_fps()
        progress = video.get_current_frame_num()
        total = video.get_frame_count()
        progress_length = round((progress / total) * bar_length)
        print('{}/{} [{}{}] {} matches'.format(
            format_time(fps, progress),
            format_time(fps, total),
            '=' * progress_length,
            ' ' * (bar_length-progress_length),
            len(outliers)
        ), end='\r')

    for video_path in args.video:
        print(video_path)
        video = Video(video_path)
        matches = recognizer.find_possible_matches(video,
            on_batch=lambda batch, outliers: print_progress(video, outliers))
        print('\nMatches:')
        for time, dist in matches:
            print(f'{format_time(1, time/1000)} ({dist})')
