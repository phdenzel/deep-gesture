"""
deep_gesture.holistic

@author: phdenzel
"""
import os
import numpy as np
import mediapipe as mp
import deep_gesture as dg


class HolisticMP(object):
    """
    Wrapper class for MediaPipe's Holistic class
    """
    N_face = 468
    N_pose = 33
    N_hand = 21
    holistic = mp.solutions.holistic
    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    holistic_styles = {
        'face_landmarks': None,
        # 'face_landmarks': drawing.DrawingSpec(color=(121, 117, 242), thickness=1, circle_radius=1),
        'face_connection': drawing_styles.get_default_face_mesh_tesselation_style(),
        # 'face_connection': drawing_styles.get_default_face_mesh_contours_style(),
        'pose_landmarks': drawing_styles.get_default_pose_landmarks_style(),
        'pose_connection': drawing.DrawingSpec(color=(107, 116, 145), thickness=2, circle_radius=2),
        'left_hand_landmarks': drawing_styles.get_default_hand_landmarks_style(),
        'left_hand_connection': drawing_styles.get_default_hand_connections_style(),
        'right_hand_landmarks': drawing_styles.get_default_hand_landmarks_style(),
        'right_hand_connection': drawing_styles.get_default_hand_connections_style(),   
    }

    def __init__(self, **kwargs):
        kwargs.setdefault('static_image_mode', False)
        kwargs.setdefault('model_complexity', 1)
        kwargs.setdefault('smooth_landmarks', True) # False
        kwargs.setdefault('enable_segmentation', False)
        kwargs.setdefault('smooth_segmentation', True)
        kwargs.setdefault('min_detection_confidence', 0.5)
        kwargs.setdefault('min_tracking_confidence', 0.5)
        self.model = self.holistic.Holistic(**kwargs)
        self.results = None

    def detection(self, image, *args):
        """
        Run the model on an image

        Args:
            image <np.ndarray> - RGB image
            args <*tuple> - dummy arguments for compatibility

        Return:
            results <mediapipe.python.solution_base.SolutionOutputs>
        """
        image.flags.writeable = False
        results = self.model.process(image)
        image.flags.writeable = True
        self.results = results
        return results

    def draw(self, image, *args, parts=(), results=None, custom_style={}):
        """
        Draw the detection results on the image

        Args:
            image <np.ndarray> - RGB image
            *args <str> - ('face' | 'pose' | 'left_hand' | 'right_hand')
        
        Kwargs:    
            results <mediapipe.python.solution_base.SolutionOutputs>
        """
        if not parts:
            parts = 'face', 'pose', 'left_hand', 'right_hand'
        if results is None:
            results = self.results
        if 'face' in parts:
            self.drawing.draw_landmarks(
                image, results.face_landmarks,
                self.holistic.FACEMESH_TESSELATION, # self.holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=self.holistic_styles['face_landmarks'],
                connection_drawing_spec=self.holistic_styles['face_connection'])
        if 'pose' in parts:
            self.drawing.draw_landmarks(
                image, results.pose_landmarks,
                self.holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.holistic_styles['pose_landmarks'],
                connection_drawing_spec=self.holistic_styles['pose_connection'])
        if 'left_hand' in parts or 'lh' in parts:
            self.drawing.draw_landmarks(
                image, results.left_hand_landmarks,
                self.holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.holistic_styles['left_hand_landmarks'],
                connection_drawing_spec=self.holistic_styles['left_hand_connection'])
        if 'right_hand' in parts or 'rh' in parts:
            self.drawing.draw_landmarks(
                image, results.right_hand_landmarks,
                self.holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.holistic_styles['left_hand_landmarks'],
                connection_drawing_spec=self.holistic_styles['left_hand_connection'])

    @property
    def landmarks(self):
        """
        A flattened array of all detected landmarks

        Return:
            landmarks <np.ndarray> - complete and holistic landmarks as
                                     flattened data array
        """
        if self.results is None:
            face_landmarks = np.zeros(self.N_face*3)
            pose_landmarks = np.zeros(self.N_pose*4)
            left_hand_landmarks = np.zeros(self.N_hand*3)
            right_hand_landmarks = np.zeros(self.N_hand*3)
            return np.concatenate([face_landmarks, pose_landmarks,
                                   left_hand_landmarks, right_hand_landmarks])
        # Face landmarks
        if self.results.face_landmarks:
            face_landmarks = np.array(
                [[p.x, p.y, p.z]
                 for p in self.results.face_landmarks.landmark]).flatten()
        else:
            face_landmarks = np.zeros(self.N_face*3)
        # Pose landmarks
        if self.results.pose_landmarks:
            pose_landmarks = np.array(
                [[p.x, p.y, p.z, p.visibility]
                 for p in self.results.pose_landmarks.landmark]).flatten()
        else:
            pose_landmarks = np.zeros(self.N_pose*4)
        # Left Hand landmarks
        if self.results.left_hand_landmarks:
            left_hand_landmarks = np.array([[p.x, p.y, p.z]
                 for p in self.results.left_hand_landmarks.landmark]).flatten()
        else:
            left_hand_landmarks = np.zeros(self.N_hand*3)
        # Right Hand landmarks
        if self.results.right_hand_landmarks:
            right_hand_landmarks = np.array([[p.x, p.y, p.z]
                 for p in self.results.right_hand_landmarks.landmark]).flatten()
        else:
            right_hand_landmarks = np.zeros(self.N_hand*3)
        return np.concatenate([face_landmarks, pose_landmarks,
                               left_hand_landmarks, right_hand_landmarks])

    def save_landmarks(self, *args, filename=None, save_dir=None, verbose=True):
        """
        Save landmarks as flattened numpy array; 
        missing landmarks are substituted with zeros

        Args:
            args <*tuple> - optional positional arguments;
                            (image, prefix, name_id, part_no, etc.)
                            see dg.utils.generate_filename

        Kwargs:
            filename <str> - name of the file
            save_dir <str> - directory in which the data is saved;
                             default: ~/.deep_gesture/tmp/
            verbose <bool> - print information to stdout

        Return:
            fpath <str> - file path where the file was saved
        """
        if args:
            args = args[1:]
        else:
            args = ('landmarks',)
        fname = dg.utils.generate_filename(*args, extension='npy') \
            if filename is None else filename
        save_dir = dg.TMP_DIR if save_dir is None else save_dir
        dg.utils.mkdir_p(save_dir)
        fpath = os.path.join(save_dir, fname)
        if os.path.exists(fpath):
            print('File already exists... rerolling filename!')
            self.save_landmarks(*args, save_dir=save_dir, verbose=verbose)
        np.save(fpath, self.landmarks)
        if verbose:
            print(fpath)
        return fpath


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from deep_gesture.camera import CVFeed
    holistic = HolisticMP()
    cvstream = CVFeed(device=0)

    cvstream.register_mod(holistic.detection, feed_passthru=True)
    cvstream.register_mod(holistic.draw, feed_passthru=True,
                          parts=('face', 'pose', 'left_hand'))
    image = cvstream.single_capture()

    plt.imshow(image)
    plt.gca().axis('off')
    plt.show()
    plt.close()
    
    
    
        
            
        
