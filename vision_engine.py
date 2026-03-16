"""
vision_engine.py

MediaPipe Face Mesh wrapper optimized for CPU edge use.
Provides landmark extraction (including iris) and returns key facial points as numpy arrays.
All processing is in-memory; no frames are written to disk.
"""
import cv2
import numpy as np
import time

# Try to support both MediaPipe solutions (classic) and the newer Tasks API.
try:
    import mediapipe as _mp
except Exception:
    _mp = None

# Landmark index groups (MediaPipe refined mesh assumptions)
_LEFT_IRIS = [468, 469, 470, 471]
_RIGHT_IRIS = [473, 474, 475, 476]

# Eye corner / eyelid / eyebrow approximate indices (widely used mediapipe references)
IDX = {
    'left_outer': 33,
    'left_inner': 133,
    'right_outer': 263,
    'right_inner': 362,
    'left_upper_eyelid': 159,
    'left_lower_eyelid': 145,
    'right_upper_eyelid': 386,
    'right_lower_eyelid': 374,
    'left_eyebrow': 105,
    'right_eyebrow': 334,
}


class VisionEngine:
    """
    VisionEngine provides a unified interface over two MediaPipe variants:
      - mp.solutions.face_mesh (classic)
      - mediapipe.tasks.python.vision FaceLandmarker (Tasks API)

    The implementation prefers `solutions` if available, otherwise falls back to `tasks`.
    """

    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self._mode = None
        self._engine = None

        # Try classic solutions API first
        if _mp is not None and hasattr(_mp, 'solutions'):
            try:
                MP_FACE = _mp.solutions.face_mesh
                self._mode = 'solutions'
                self._engine = MP_FACE.FaceMesh(
                    static_image_mode=static_image_mode,
                    max_num_faces=max_num_faces,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                return
            except Exception:
                self._engine = None

        # Fallback to Tasks API (FaceLandmarker) if available
        try:
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, BaseOptions, VisionRunningMode

            self._mode = 'tasks'
            # Attempt to use bundled task file name; if it's not present, this will raise at create time
            base_options = mp_vision.BaseOptions(model_asset_path='face_landmarker.task')
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.VisionRunningMode.VIDEO,
                output_face_meshes=True,
                output_iris_landmarks=True,
            )
            self._engine = mp_vision.FaceLandmarker.create_from_options(options)
            # Save refs for runtime usage
            self._mp_vision = mp_vision
            return
        except Exception:
            # If neither is available, surface helpful error when processing
            self._mode = None
            self._engine = None

        # Final fallback: simple OpenCV Haar cascade face detector to keep prototype running.
        if self._engine is None:
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
                if self._face_cascade.empty():
                    self._face_cascade = None
                else:
                    self._mode = 'haar'
            except Exception:
                self._face_cascade = None
        # Debug: report selected mode
        try:
            print(f'VisionEngine selected mode: {self._mode}')
        except Exception:
            pass

    def _landmarks_to_np(self, landmarks, w, h):
        """Convert a sequence of landmarks (with .x/.y) to Nx2 numpy array in pixel coords."""
        pts = np.zeros((len(landmarks), 2), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            pts[i, 0] = lm.x * w
            pts[i, 1] = lm.y * h
        return pts

    def process(self, frame):
        """
        Process a BGR frame and return a dict with pixel landmarks and selected key points.
        Returns None if no face detected.
        """
        # If no engine is available, allow 'haar' fallback to proceed; otherwise error.
        if self._engine is None and getattr(self, '_mode', None) != 'haar':
            raise RuntimeError('No MediaPipe engine available (solutions or tasks).')

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Solutions API path
        if self._mode == 'solutions':
            results = self._engine.process(rgb)
            if not results or not getattr(results, 'multi_face_landmarks', None):
                return None
            face = results.multi_face_landmarks[0]
            lm = self._landmarks_to_np(face.landmark, w, h)

        # Tasks API path
        elif self._mode == 'tasks':
            try:
                # create a mediapipe Image from the numpy array
                mp_image = None
                # Try a few constructors depending on mediapipe version
                if hasattr(self._mp_vision.Image, 'create_from_rgb_image'):
                    mp_image = self._mp_vision.Image.create_from_rgb_image(rgb)
                elif hasattr(self._mp_vision.Image, 'create_from_array'):
                    mp_image = self._mp_vision.Image.create_from_array(rgb)
                else:
                    # last resort: try TensorImage
                    if hasattr(self._mp_vision, 'TensorImage'):
                        mp_image = self._mp_vision.TensorImage.create_from_array(rgb)

                if mp_image is None:
                    raise RuntimeError('Cannot construct mediapipe Image from numpy array for Tasks API.')

                # timestamp in ms
                ts = int(time.time() * 1000)
                results = self._engine.detect_for_video(mp_image, ts)

                faces = getattr(results, 'face_landmarks', None) or getattr(results, 'face_landmarks_list', None)
                if not faces:
                    return None

                first = faces[0]
                # first may be a LandmarkList or similar; try common attr names
                if hasattr(first, 'landmark'):
                    lm_seq = first.landmark
                elif hasattr(first, 'landmarks'):
                    lm_seq = first.landmarks
                else:
                    # maybe it's already a list
                    lm_seq = first

                lm = self._landmarks_to_np(lm_seq, w, h)

            except Exception as e:
                raise RuntimeError(f'Failed to run MediaPipe Tasks FaceLandmarker: {e}')

        else:
            # Haar fallback or none
            if getattr(self, '_mode', None) == 'haar' and getattr(self, '_face_cascade', None) is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(faces) == 0:
                    return None
                x, y, fw, fh = faces[0]
                # approximate landmarks array (478,2) to keep index usage consistent
                lm = np.zeros((478, 2), dtype=np.float32)
                # approximate left/right iris centers
                left_eye_center = (x + int(fw * 0.3), y + int(fh * 0.45))
                right_eye_center = (x + int(fw * 0.7), y + int(fh * 0.45))
                lm[_LEFT_IRIS, :] = np.array([[left_eye_center[0], left_eye_center[1]]] * len(_LEFT_IRIS))
                lm[_RIGHT_IRIS, :] = np.array([[right_eye_center[0], right_eye_center[1]]] * len(_RIGHT_IRIS))
                # set corners/eyebrows/eyelids using proportional positions
                lm[IDX['left_inner']] = np.array([x + int(fw * 0.35), y + int(fh * 0.45)])
                lm[IDX['left_outer']] = np.array([x + int(fw * 0.15), y + int(fh * 0.45)])
                lm[IDX['right_inner']] = np.array([x + int(fw * 0.65), y + int(fh * 0.45)])
                lm[IDX['right_outer']] = np.array([x + int(fw * 0.85), y + int(fh * 0.45)])
                lm[IDX['left_upper_eyelid']] = np.array([x + int(fw * 0.35), y + int(fh * 0.40)])
                lm[IDX['right_upper_eyelid']] = np.array([x + int(fw * 0.65), y + int(fh * 0.40)])
                lm[IDX['left_eyebrow']] = np.array([x + int(fw * 0.35), y + int(fh * 0.30)])
                lm[IDX['right_eyebrow']] = np.array([x + int(fw * 0.65), y + int(fh * 0.30)])
                # continue to build keypoints from lm
            else:
                return None

        # Iris centers (mean of iris landmark group) — vectorized when available
        try:
            left_iris = lm[_LEFT_IRIS, :].mean(axis=0)
            right_iris = lm[_RIGHT_IRIS, :].mean(axis=0)
        except Exception:
            left_iris = lm[0]
            right_iris = lm[-1]

        keypoints = {
            'landmarks': lm,
            'left_iris': left_iris,
            'right_iris': right_iris,
            'left_outer': lm[IDX['left_outer']],
            'left_inner': lm[IDX['left_inner']],
            'right_outer': lm[IDX['right_outer']],
            'right_inner': lm[IDX['right_inner']],
            'left_upper_eyelid': lm[IDX['left_upper_eyelid']],
            'left_lower_eyelid': lm[IDX['left_lower_eyelid']],
            'right_upper_eyelid': lm[IDX['right_upper_eyelid']],
            'right_lower_eyelid': lm[IDX['right_lower_eyelid']],
            'left_eyebrow': lm[IDX['left_eyebrow']],
            'right_eyebrow': lm[IDX['right_eyebrow']],
            'image_size': (w, h),
        }
        return keypoints

    def close(self):
        if self._engine is None:
            return
        try:
            self._engine.close()
        except Exception:
            pass
