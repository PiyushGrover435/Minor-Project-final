"""
sentin_edge.py

Minimal prototype runner — demonstrates VisionEngine iris tracking with OpenCV overlay.
"""
import cv2
from vision_engine import VisionEngine


def run():
    engine = VisionEngine()
    cap = cv2.VideoCapture(0)

    print('Starting Sentin-Edge AI Prototype (VisionEngine)...')
    print("Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                kp = engine.process(frame)
            except Exception as e:
                print('VisionEngine error:', e)
                break

            if kp is not None:
                try:
                    li = tuple(kp['left_iris'].astype(int))
                    cv2.circle(frame, li, 4, (0, 255, 0), -1)
                except Exception:
                    pass

            cv2.putText(frame, 'Sentin-Edge AI: ACTIVE', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, 'Privacy: On-Device (In-Memory)', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Phase 1 Prototype - Sentin-Edge', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        engine.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run()