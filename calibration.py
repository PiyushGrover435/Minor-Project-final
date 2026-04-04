import cv2
import time
import numpy as np

class CalibrationEngine:
    """
    Manages the 'Point Chasing' calibration phase for the Spatial-Temporal Gaze Head.
    Displays a sequence of points (Center, Top-Left, Top-Right, Bottom-Left, Bottom-Right).
    Collects eye patches and fine-tunes the local TCN model via SGD.
    """
    def __init__(self, screen_w=640, screen_h=480, points_count=5, duration_per_point=3.0):
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        # 5 standard calibration points (normalized coords)
        # Margin added so it's not strictly at the edge
        m_x, m_y = 0.1, 0.1 
        self.targets_norm = [
            (0.5, 0.5),             # Center
            (m_x, m_y),             # Top-Left
            (1.0 - m_x, m_y),       # Top-Right
            (m_x, 1.0 - m_y),       # Bottom-Left
            (1.0 - m_x, 1.0 - m_y)  # Bottom-Right
        ]
        
        self.duration = duration_per_point
        self.active = False
        self.current_idx = 0
        self.point_start_time = 0
        self.bg_color = (128, 128, 128) # Neutral grey for privacy-preservation 2025 standard
        
    def start(self):
        self.active = True
        self.current_idx = 0
        self.point_start_time = time.time()
        print("\n[Calibration] Phase Started. Please follow the red dot.")
        
    def stop(self):
        self.active = False
        print("[Calibration] Phase Completed. Local Gaze model updated!\n")
        
    def get_current_target(self):
        """Returns normalized coordinates of current target."""
        if not self.active or self.current_idx >= len(self.targets_norm):
            return None
        return self.targets_norm[self.current_idx]
        
    def update_and_draw(self, frame_size, kp, frame, gaze_head):
        """
        Updates state time. If active, overrides the frame with the calibration screen.
        Calls fine_tune on the gaze_head while the user stares at the point.
        """
        if not self.active:
            return frame, False
            
        now = time.time()
        elapsed = now - self.point_start_time
        
        if elapsed > self.duration:
            self.current_idx += 1
            self.point_start_time = now
            if self.current_idx >= len(self.targets_norm):
                self.stop()
                gaze_head.save_weights()
                return frame, False
                
        # 1. Drive the learning
        # We only pass frames for SGD after 0.5s to ensure the user has moved their eyes to the dot
        target_norm = self.targets_norm[self.current_idx]
        if elapsed > 0.5 and kp is not None:
            # We pass the real camera frame to fine_tune so it extracts the actual eye patch
            loss = gaze_head.fine_tune(kp, frame, target_norm[0], target_norm[1])
            
        # 2. Render the neutral calibration UI OVER the camera frame
        h, w = frame_size
        canvas = np.full((h, w, 3), self.bg_color, dtype=np.uint8)
        
        # Calculate pixel position for the red dot
        tx, ty = target_norm
        px, py = int(tx * w), int(ty * h)
        
        # Animating the circle radius (pulse effect) to draw attention
        pulse = int(5 * np.sin(elapsed * 10))
        cv2.circle(canvas, (px, py), 15 + pulse, (0, 0, 255), -1)
        cv2.circle(canvas, (px, py), 15 + pulse, (255, 255, 255), 2)
        
        cv2.putText(canvas, f"Calibration Point {self.current_idx + 1}/{len(self.targets_norm)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        # Fade transition logic (fade out point during last 0.2s)
        if elapsed > self.duration - 0.2:
            alpha = (self.duration - elapsed) / 0.2
            canvas = cv2.addWeighted(np.full_like(canvas, self.bg_color), 1-alpha, canvas, alpha, 0)
            
        return canvas, True
