"""
landmark_smoother.py

Per-landmark Kalman filter for stabilising MediaPipe iris and keypoint coordinates.
Slots between vision_engine.process() and keypoint dict consumption.
No frames are stored — state is purely numerical.
"""
import numpy as np

class LandmarkKalmanFilter:
    """
    Scalar 1D Kalman filter with a constant-velocity model.
    One instance per coordinate axis per landmark (x and y).

    State vector: [position, velocity]
    """
    def __init__(self, process_noise=1e-3, measurement_noise=1e-1):
        self.Q = process_noise       # how much the true position can jump per frame
        self.R = measurement_noise   # how noisy the webcam measurement is

        self.x = 0.0   # position estimate
        self.v = 0.0   # velocity estimate
        self.p = 1.0   # error covariance (position)
        self.pv = 0.0  # cross covariance
        self.vv = 1.0  # velocity covariance
        self._initialized = False

    def update(self, measurement: float) -> float:
        if not self._initialized:
            self.x = measurement
            self._initialized = True
            return self.x

        # ── Predict ──
        x_pred = self.x + self.v
        p_pred = self.p + self.vv + 2 * self.pv + self.Q

        # ── Update (measurement model: observe position only) ──
        K = p_pred / (p_pred + self.R)   # Kalman gain
        innovation = measurement - x_pred

        self.x = x_pred + K * innovation
        self.v = self.v + 0.1 * K * innovation   # simple velocity correction
        self.p = (1 - K) * p_pred

        return self.x


class KeypointSmoother:
    """
    Applies independent Kalman filters to each named keypoint (x and y axes).
    Usage:
        smoother = KeypointSmoother()
        smooth_kp = smoother.smooth(raw_keypoints)
    """
    SMOOTH_KEYS = (
        'left_iris', 'right_iris',
        'left_inner', 'left_outer',
        'right_inner', 'right_outer',
        'left_upper_eyelid', 'left_lower_eyelid',
        'right_upper_eyelid', 'right_lower_eyelid',
        'left_eyebrow', 'right_eyebrow',
    )

    def __init__(self, process_noise=1e-3, measurement_noise=0.08):
        self._filters = {}
        self.pn = process_noise
        self.mn = measurement_noise

    def _get_filter(self, key: str, axis: int) -> LandmarkKalmanFilter:
        fid = (key, axis)
        if fid not in self._filters:
            self._filters[fid] = LandmarkKalmanFilter(self.pn, self.mn)
        return self._filters[fid]

    def smooth(self, keypoints: dict) -> dict:
        """Return a copy of keypoints with smoothed coordinates for tracked keys."""
        smoothed = dict(keypoints)
        for key in self.SMOOTH_KEYS:
            if key not in keypoints:
                continue
            pt = np.asarray(keypoints[key], dtype=np.float32)
            sx = self._get_filter(key, 0).update(float(pt[0]))
            sy = self._get_filter(key, 1).update(float(pt[1]))
            # Preserve original array shape (2D pixel coords)
            new_pt = pt.copy()
            new_pt[0] = sx
            new_pt[1] = sy
            smoothed[key] = new_pt
        return smoothed

    def reset(self):
        """Call when face tracking is lost (kp is None) to avoid stale velocity."""
        self._filters.clear()

