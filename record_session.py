"""
record_session.py

Records a live webcam session to CSV for benchmarking.
Captures raw MediaPipe landmarks (478 x 3) per frame — no images saved.

Usage:
  python record_session.py --output_csv mpiigaze_landmarks.csv --max_frames 500
  (Press 'q' to stop early)
"""
import argparse
import csv
import time
import cv2
from vision_engine import VisionEngine


def main():
    p = argparse.ArgumentParser(description='Record live landmarks to CSV for benchmark.')
    p.add_argument('--output_csv', default='mpiigaze_landmarks.csv', help='Output CSV path')
    p.add_argument('--max_frames', type=int, default=500, help='Max frames to capture')
    args = p.parse_args()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[ERROR] Cannot open camera.')
        return

    engine = VisionEngine()
    rows_written = 0

    print(f'Recording up to {args.max_frames} frames to {args.output_csv}')
    print("Look around naturally — Center, Left, Right — to generate varied data.")
    print("Press 'q' to stop early.\n")

    with open(args.output_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)

        while rows_written < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            kp = engine.process(frame)

            if kp is not None and 'landmarks' in kp:
                lm = kp['landmarks']  # Nx2 array
                # Build row: label + flattened landmarks (x, y, 0.0 for z placeholder)
                label = f'frame_{rows_written:05d}'
                flat = []
                for i in range(lm.shape[0]):
                    flat.extend([lm[i, 0], lm[i, 1], 0.0])
                writer.writerow([label] + flat)
                rows_written += 1

                # Show progress overlay
                cv2.putText(frame, f'Recording: {rows_written}/{args.max_frames}',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No face detected...', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Sentin-Edge: Recording Session', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    engine.close()
    cv2.destroyAllWindows()
    print(f'\nDone. Wrote {rows_written} landmark rows to {args.output_csv}')


if __name__ == '__main__':
    main()
