"""
calibration.py

Enhanced 5-point gaze calibration with visual countdown, progress bar,
animated target rings, settling indicator, and step instructions.
"""
import cv2
import time
import numpy as np


# ── Visual Constants ────────────────────────────────────────────────────
_BG_DARK      = (25, 25, 30)       # Dark background (modern look)
_COL_TARGET   = (80, 140, 255)     # Warm orange-blue target (BGR)
_COL_RING     = (120, 180, 255)    # Outer ring glow
_COL_GREEN    = (100, 220, 100)    # Settling / progress fill
_COL_DIMTEXT  = (150, 150, 160)    # Muted text
_COL_WHITE    = (240, 240, 245)    # Crisp white
_COL_BAR_BG   = (50, 50, 55)      # Progress bar background
_COL_BAR_FILL = (80, 200, 120)     # Progress bar fill
_COL_SETTLING = (60, 60, 180)      # Orange for settling phase

_TARGET_LABELS = ['Center', 'Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right']


class CalibrationEngine:
    """
    Manages the 'Point Chasing' calibration phase for the Spatial-Temporal Gaze Head.
    Displays a sequence of points (Center, Top-Left, Top-Right, Bottom-Left, Bottom-Right).
    Collects eye patches and fine-tunes the local TCN model via SGD.

    Enhanced with: countdown timer, settling indicator, progress bar,
    animated concentric rings, and step-by-step instructions.
    """

    def __init__(self, screen_w=640, screen_h=480, points_count=5, duration_per_point=3.0):
        self.screen_w = screen_w
        self.screen_h = screen_h

        # 5 standard calibration points (normalized coords)
        m_x, m_y = 0.12, 0.12
        self.targets_norm = [
            (0.5, 0.5),               # Center
            (m_x, m_y),               # Top-Left
            (1.0 - m_x, m_y),         # Top-Right
            (m_x, 1.0 - m_y),         # Bottom-Left
            (1.0 - m_x, 1.0 - m_y),   # Bottom-Right
        ]

        self.settle_time    = 0.6     # seconds to wait before learning
        self.duration       = duration_per_point
        self.active         = False
        self.current_idx    = 0
        self.point_start_time = 0
        self._loss_accum    = []      # collect loss values per point

    def start(self):
        self.active = True
        self.current_idx = 0
        self.point_start_time = time.time()
        self._loss_accum = []
        print("\n[Calibration] Phase Started. Follow the target dot through 5 screen positions.")

    def stop(self):
        self.active = False
        print("[Calibration] Phase Completed. Local Gaze model updated!\n")

    def get_current_target(self):
        """Returns normalized coordinates of current target."""
        if not self.active or self.current_idx >= len(self.targets_norm):
            return None
        return self.targets_norm[self.current_idx]

    # ── Drawing helpers ─────────────────────────────────────────────────

    @staticmethod
    def _draw_panel(canvas, x, y, w, h, color=(40, 40, 45), alpha=0.75):
        """Draw a semi-transparent rounded rectangle panel."""
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    @staticmethod
    def _draw_progress_bar(canvas, x, y, w, h, progress, bg=_COL_BAR_BG, fill=_COL_BAR_FILL):
        """Draw a rounded progress bar."""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
        fill_w = int(w * max(0.0, min(1.0, progress)))
        if fill_w > 0:
            cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), fill, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 85), 1)

    def _draw_target(self, canvas, px, py, elapsed, is_settling):
        """Draw animated concentric target rings with pulse effect."""
        t = elapsed * 2.5  # animation speed

        # Outer expanding ring (fades out)
        ring_r = 30 + int(8 * np.sin(t))
        ring_alpha = int(80 + 40 * np.sin(t * 0.7))
        cv2.circle(canvas, (px, py), ring_r, (*_COL_RING[:2], ring_alpha), 2, cv2.LINE_AA)

        # Middle ring
        mid_r = 20 + int(4 * np.sin(t + 1.0))
        cv2.circle(canvas, (px, py), mid_r, _COL_RING, 1, cv2.LINE_AA)

        # Core dot
        core_col = _COL_SETTLING if is_settling else _COL_GREEN
        core_r = 10 + int(3 * np.sin(t * 1.5))
        cv2.circle(canvas, (px, py), core_r, core_col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), core_r, _COL_WHITE, 1, cv2.LINE_AA)

        # Crosshair lines (subtle)
        line_len = 45
        line_col = (60, 60, 65)
        cv2.line(canvas, (px - line_len, py), (px - core_r - 5, py), line_col, 1, cv2.LINE_AA)
        cv2.line(canvas, (px + core_r + 5, py), (px + line_len, py), line_col, 1, cv2.LINE_AA)
        cv2.line(canvas, (px, py - line_len), (px, py - core_r - 5), line_col, 1, cv2.LINE_AA)
        cv2.line(canvas, (px, py + core_r + 5), (px, py + line_len), line_col, 1, cv2.LINE_AA)

    # ── Main update ─────────────────────────────────────────────────────

    def update_and_draw(self, frame_size, kp, frame, gaze_head):
        """
        Updates state. If active, overrides frame with calibration screen.
        Calls fine_tune on the gaze_head while user fixates on the target.
        """
        if not self.active:
            return frame, False

        now = time.time()
        elapsed = now - self.point_start_time

        if elapsed > self.duration:
            self.current_idx += 1
            self.point_start_time = now
            elapsed = 0.0
            self._loss_accum = []
            if self.current_idx >= len(self.targets_norm):
                self.stop()
                gaze_head.save_weights()
                return frame, False

        # 1. Drive the learning (only after settling time)
        target_norm = self.targets_norm[self.current_idx]
        is_settling = elapsed < self.settle_time
        if not is_settling and kp is not None:
            loss = gaze_head.fine_tune(kp, frame, target_norm[0], target_norm[1])
            if loss is not None:
                self._loss_accum.append(float(loss))

        # 2. Render the calibration UI
        h, w = frame_size
        canvas = np.full((h, w, 3), _BG_DARK, dtype=np.uint8)

        # ── Target dot ──────────────────────────────────────────────
        tx, ty = target_norm
        px, py = int(tx * w), int(ty * h)
        self._draw_target(canvas, px, py, elapsed, is_settling)

        # ── Header panel ────────────────────────────────────────────
        self._draw_panel(canvas, 0, 0, w, 72, (20, 20, 25), 0.85)

        # Title
        cv2.putText(canvas, 'GAZE CALIBRATION', (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COL_WHITE, 2, cv2.LINE_AA)

        # Step indicator: "Point 2 / 5  —  Top-Left"
        step = self.current_idx + 1
        total = len(self.targets_norm)
        label = _TARGET_LABELS[self.current_idx] if self.current_idx < len(_TARGET_LABELS) else '?'
        cv2.putText(canvas, f'Point {step} / {total}  |  {label}', (15, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COL_DIMTEXT, 1, cv2.LINE_AA)

        # Overall progress bar (across all points)
        overall = (self.current_idx + elapsed / self.duration) / total
        bar_y = 64
        self._draw_progress_bar(canvas, 280, bar_y - 8, w - 295, 10, overall)

        # ── Per-point countdown ─────────────────────────────────────
        remaining = max(0.0, self.duration - elapsed)
        countdown_str = f'{remaining:.1f}s'

        # Position countdown near the target
        cd_x = max(15, min(px - 20, w - 80))
        cd_y = max(90, py - 55)
        cv2.putText(canvas, countdown_str, (cd_x, cd_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, _COL_WHITE, 2, cv2.LINE_AA)

        # ── Per-point progress ring ─────────────────────────────────
        point_progress = elapsed / self.duration
        angle = int(360 * point_progress)
        cv2.ellipse(canvas, (px, py), (38, 38), -90, 0, angle, _COL_BAR_FILL, 2, cv2.LINE_AA)

        # ── Status text (settling vs learning) ──────────────────────
        status_y = h - 55
        self._draw_panel(canvas, 0, status_y - 10, w, 65, (20, 20, 25), 0.85)

        if is_settling:
            cv2.putText(canvas, 'SETTLING — Move your eyes to the target...',
                        (15, status_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        _COL_SETTLING, 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, 'LEARNING — Hold your gaze steady on the dot',
                        (15, status_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        _COL_GREEN, 1, cv2.LINE_AA)

        # Loss readout
        if self._loss_accum:
            avg_loss = sum(self._loss_accum) / len(self._loss_accum)
            cv2.putText(canvas, f'Avg Loss: {avg_loss:.4f}  ({len(self._loss_accum)} samples)',
                        (15, status_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        _COL_DIMTEXT, 1, cv2.LINE_AA)

        # ── Fade transition during last 0.25s ──────────────────────
        if elapsed > self.duration - 0.25:
            alpha = (self.duration - elapsed) / 0.25
            bg = np.full_like(canvas, _BG_DARK)
            canvas = cv2.addWeighted(bg, 1 - alpha, canvas, alpha, 0)

        return canvas, True
