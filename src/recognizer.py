import math
import numpy as np
import cv2

class CelRecognizer:
    def __init__(self, cel, expected_size):
        """CelRecognizer contstructor

        Arguments:
            cel: A numpy array containing the cel image
            expected_size: Frames that are passed to this recognizer should be roughly this
                size for good results. (width, weight)
        """
        cel_ar = cel.shape[1] // cel.shape[0]
        self._cel = cv2.resize(cel, (expected_size[0], expected_size[1] * cel_ar))
        self._orb = cv2.ORB_create()
        self._cel_kp, self._cel_desc = self._orb.detectAndCompute(self._cel, None)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
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
    
    def _find_outliers(self, results):
        """Finds outliers in a list of batches.
        Outliers are considered to be points where the value is more than 4 standard deviations below the mean.
        
        Arguments:
            results: An array of tuples, where the first item is an arbitrary label and the second item is the
                value to find outliers over.
        """
        # Get the mean and standard deviation of the distances
        mean = np.mean(results, axis=0)[1]
        std = np.std(results, axis=0)[1]
        # Note: the lack of an absolute value in finding the deviance from the mean is intentional.
        # We only care about outliers that are below the mean, as those are the ones that could be matches.
        return [ (i, dist) for i, dist in results if mean - dist > 4 * std ]
    
    def find_possible_matches(self, frame_iterator, batch_size=24, on_batch=None):
        """Finds possible uses of the cel in a video.

        Arguments:
            frame_iterator: An iterator over the frames of the video. Each item yielded by this iterator should
                be a numpy array.
            batch_size: The results are grouped into batches of this size when looking for matches.
            on_batch: A callback that is called after every batch. Two arguments are passed to this function:
                batch: A tuple containing the index of the first frame in the batch and the minimum distance among
                    the frames in the batch
                matches: An array of batches that might be matches.

        Returns an array of batches that might be matches.
        """
        results = []
        batch_start_frame = 0
        batch_min_dist = math.inf
        for frame_num, frame in enumerate(frame_iterator):
            # Add the distance for this frame to the batch
            batch_min_dist = min(batch_min_dist, self._distance(frame))

            if frame_num - batch_start_frame >= batch_size:
                # Add the batch to the results (unless all of the distances were infinite)
                if batch_min_dist != math.inf:
                    results.append((batch_start_frame, batch_min_dist))

                    # Run the on_batch callback if given
                    if on_batch is not None:
                        on_batch((batch_start_frame, batch_min_dist), self._find_outliers(results))
                
                # Start a new batch
                batch_start_frame = frame_num + 1
                batch_min_dist = math.inf

        
        # Add the last batch to the results
        if batch_min_dist != math.inf:
            results.append((batch_start_frame, batch_min_dist))

        # Return all of the batches that were outliers below the mean, as those are the
        # ones that might be a match
        return self._find_outliers(results)

def frames(video_file):
    video = cv2.VideoCapture(video_file)
    while True:
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield frame
        else:
            break
    video.release()


if __name__ == '__main__':
    data_dir = 'C:/Users/Noah/Documents/se_data'
    cel_path = data_dir + '/cels/Lion-O - Sword In A Hole/Lion-O.jpg'
    video_path = data_dir + '/videos/Thundercats/Thundercats_S1_Ep35_–_Sword_in_a_Hole.mp4'

    fps = 23.98
    def format_time(frame_num):
        s = frame_num / fps
        m = s // 60
        s = math.floor(s % 60)
        return f'{int(m)}:{s:02d}'

    cel = cv2.cvtColor(cv2.imread(cel_path), cv2.COLOR_BGR2GRAY)
    recognizer = CelRecognizer(cel, (480, 360))

    matches = recognizer.find_possible_matches(frames(video_path), on_batch=print)
    print(matches)
