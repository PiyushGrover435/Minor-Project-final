"""
vision_engine.py

MediaPipe Face Mesh wrapper optimized for CPU edge use.
Provides landmark extraction (including iris) and returns key facial points as numpy arrays.
All processing is in-memory; no frames are written to disk.
"""
import cv2
import numpy as np
import time

from landmark_indices import LEFT_IRIS, RIGHT_IRIS, IDX, LEFT_EYE_POINTS, RIGHT_EYE_POINTS

# Try to support both MediaPipe solutions (classic) and the newer Tasks API.
try:
    import mediapipe as _mp
except Exception:
    _mp = None


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

        # ── Try classic solutions API first ─────────────────────────────
        if _mp is not None and hasattr(_mp, 'solutions'):
            try:
                mp_face = _mp.solutions.face_mesh
                self._engine = mp_face.FaceMesh(
                    static_image_mode=static_image_mode,
                    max_num_faces=max_num_faces,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self._mode = 'solutions'
                return
            except Exception:
                self._engine = None

        # ── Fallback to Tasks API (FaceLandmarker) ──────────────────────
        try:
            import mediapipe as mp
            
            base_options = mp.tasks.BaseOptions(model_asset_path='face_landmarker.task')
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
            )
            self._engine = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
            self._mode = 'tasks'
            return
        except Exception as e:
            print("[VisionEngine] Tasks init failed:", e)
            self._mode = None
            self._engine = None

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _landmarks_to_np(landmarks, w, h):
        """Convert a sequence of landmarks (with .x/.y) to Nx2 numpy array in pixel coords."""
        pts = np.empty((len(landmarks), 2), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            pts[i, 0] = lm.x * w
            pts[i, 1] = lm.y * h
        return pts

    # ── Public API ──────────────────────────────────────────────────────

    def process(self, frame):
        """
        Process a BGR frame and return a dict with pixel landmarks and selected key points.
        Returns None if no face detected.
        """
        if self._engine is None:
            raise RuntimeError('No MediaPipe engine available (solutions or tasks).')

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Solutions API path ──────────────────────────────────────────
        if self._mode == 'solutions':
            results = self._engine.process(rgb)
            if not results or not getattr(results, 'multi_face_landmarks', None):
                return None
            lm = self._landmarks_to_np(results.multi_face_landmarks[0].landmark, w, h)

        # ── Tasks API path ──────────────────────────────────────────────
        elif self._mode == 'tasks':
            try:
                mp_image = None
                for ctor in ('create_from_rgb_image', 'create_from_array'):
                    fn = getattr(self._mp.Image, ctor, None)
                    if fn is not None:
                        mp_image = fn(rgb)
                        break
                if mp_image is None and hasattr(self._mp, 'TensorImage'):
                    mp_image = getattr(self._mp, 'TensorImage').create_from_array(rgb)
                if mp_image is None:
                    try:
                        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
                    except Exception:
                        pass
                if mp_image is None:
                    raise RuntimeError('Cannot construct mediapipe Image for Tasks API.')

                ts = int(time.time() * 1000)
                results = self._engine.detect_for_video(mp_image, ts)
                faces = getattr(results, 'face_landmarks', None) or \
                        getattr(results, 'face_landmarks_list', None)
                if not faces:
                    return None

                first = faces[0]
                lm_seq = getattr(first, 'landmark', None) or \
                         getattr(first, 'landmarks', None) or first
                lm = self._landmarks_to_np(lm_seq, w, h)

            except Exception as e:
                raise RuntimeError(f'Failed to run MediaPipe Tasks FaceLandmarker: {e}')
        else:
            return None

        # ── Extract keypoints ───────────────────────────────────────────
        # Iris centres (mean of N iris landmarks per eye)
        try:
            left_iris  = lm[LEFT_IRIS, :].mean(axis=0)
            right_iris = lm[RIGHT_IRIS, :].mean(axis=0)
        except (IndexError, KeyError):
            left_iris  = lm[0]
            right_iris = lm[-1]

        keypoints = {
            'landmarks':           lm,
            'left_iris':           left_iris,
            'right_iris':          right_iris,
            'image_size':          (w, h),
            'face_bbox':           (
                int(np.min(lm[:, 0])), # x_min
                int(np.min(lm[:, 1])), # y_min
                int(np.max(lm[:, 0])), # x_max
                int(np.max(lm[:, 1]))  # y_max
            ),
            'left_eye_points':     tuple(lm[idx] for idx in LEFT_EYE_POINTS),
            'right_eye_points':    tuple(lm[idx] for idx in RIGHT_EYE_POINTS),
            'left_iris_pts':       tuple(lm[idx] for idx in LEFT_IRIS),
            'right_iris_pts':      tuple(lm[idx] for idx in RIGHT_IRIS)
        }
        
        # ── Head Pose Estimation (6-DOF via solvePnP) ──────────────────
        #
        # Canonical 3D face model points (generic human face proportions).
        # These correspond to MediaPipe landmark indices:
        #   1  = Nose tip
        #   152 = Chin
        #   33  = Left eye outer corner
        #   263 = Right eye outer corner
        #   61  = Left mouth corner
        #   291 = Right mouth corner
        _MODEL_3D = np.array([
            (0.0,    0.0,    0.0),      # Nose tip
            (0.0,   -63.6,  -12.5),     # Chin
            (-43.3,  32.7,  -26.0),     # Left eye outer
            (43.3,   32.7,  -26.0),     # Right eye outer
            (-28.9, -28.9,  -24.1),     # Left mouth
            (28.9,  -28.9,  -24.1),     # Right mouth
        ], dtype=np.float64)
        _POSE_IDX = [1, 152, 33, 263, 61, 291]

        try:
            image_pts = np.array([lm[i] for i in _POSE_IDX], dtype=np.float64)  # Nx2 pixel

            # Approximate camera intrinsics from frame dimensions
            focal = float(w)
            cx_cam, cy_cam = w / 2.0, h / 2.0
            cam_matrix = np.array([
                [focal, 0,     cx_cam],
                [0,     focal, cy_cam],
                [0,     0,     1.0   ],
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            success, rvec, tvec = cv2.solvePnP(
                _MODEL_3D, image_pts, cam_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if success:
                # rvec is Rodrigues (3x1).  Convert to degrees for readability.
                pitch = float(np.degrees(rvec[0, 0]))  # Rx — nodding
                yaw   = float(np.degrees(rvec[1, 0]))  # Ry — shaking head
                roll  = float(np.degrees(rvec[2, 0]))  # Rz — tilting head
                tx, ty, tz = float(tvec[0, 0]), float(tvec[1, 0]), float(tvec[2, 0])
                keypoints['head_pose'] = (pitch, yaw, roll)
                keypoints['head_translation'] = (tx, ty, tz)
            else:
                keypoints['head_pose'] = (0.0, 0.0, 0.0)
                keypoints['head_translation'] = (0.0, 0.0, 0.0)
        except Exception:
            keypoints['head_pose'] = (0.0, 0.0, 0.0)
            keypoints['head_translation'] = (0.0, 0.0, 0.0)

        # Add all named indices from the canonical map
        for name, idx in IDX.items():
            keypoints[name] = lm[idx]

        return keypoints

    def close(self):
        if self._engine is None:
            return
        try:
            self._engine.close()
        except Exception:
            pass
