#!/usr/bin/env python3
"""infer_from_landmarks.py

Read CSV produced by `preprocess_data.py` and run the analytics pipeline
to produce gaze, stress, integrity per sample. Outputs JSONL with results.

Usage:
  python infer_from_landmarks.py --input_csv mpiigaze_landmarks.csv --output_jsonl results.jsonl

The CSV expected format: first column = label (filename or class), then 478*3 float columns (x,y,z) in normalized coords.
"""
import argparse
import csv
import json
from analytics import RealtimeAnalyzer, compute_gaze, compute_stress

NUM_LM = 478

def row_to_keypoints(row):
    # row[0] is label; rest are floats
    vals = row[1:1+NUM_LM*3]
    try:
        vals = [float(x) if x != '' else 0.0 for x in vals]
    except Exception:
        vals = [0.0]*(NUM_LM*3)

    pts = [(vals[i], vals[i+1], vals[i+2]) for i in range(0, len(vals), 3)]

    # Map approximate indices to the semantic keys used by analytics.
    # We attempt to follow MediaPipe refined mesh indices; if not available, fall back to reasonable guesses.
    def safe(i):
        if i < 0 or i >= len(pts):
            return (0.0, 0.0, 0.0)
        return pts[i]

    keypoints = {
        'left_iris': safe(468),
        'right_iris': safe(473),
        'left_inner': safe(133),
        'left_outer': safe(33),
        'right_inner': safe(362),
        'right_outer': safe(263),
        'left_eyebrow': safe(70),
        'right_eyebrow': safe(300),
        'left_upper_eyelid': safe(159),
        'left_lower_eyelid': safe(145),
        'right_upper_eyelid': safe(386),
        'right_lower_eyelid': safe(374),
    }
    return keypoints


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_jsonl", required=True)
    args = p.parse_args()

    analyzer = RealtimeAnalyzer(window=12, calib_frames=10)

    with open(args.input_csv, newline='', encoding='utf-8') as fh_in, open(args.output_jsonl, 'w', encoding='utf-8') as fh_out:
        reader = csv.reader(fh_in)
        for row in reader:
            if not row:
                continue
            label = row[0]
            keypoints = row_to_keypoints(row)
            gaze_label, gaze_vals, stress_level, stress_score, integrity = analyzer.update(keypoints)
            out = {
                'label': label,
                'gaze': gaze_label,
                'gaze_vals': gaze_vals,
                'stress_level': stress_level,
                'stress_score': stress_score,
                'integrity': integrity,
            }
            fh_out.write(json.dumps(out) + "\n")

    print("Done. Wrote results to", args.output_jsonl)


if __name__ == '__main__':
    main()
