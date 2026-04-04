#!/usr/bin/env python3
"""infer_from_landmarks.py

Read CSV produced by `preprocess_data.py` and run the analytics pipeline
to produce gaze, stress, integrity per sample.  Outputs JSONL with results.

Usage:
  python infer_from_landmarks.py --input_csv mpiigaze_landmarks.csv --output_jsonl results.jsonl

The CSV expected format:
  first column = label (filename or class),
  then 478*3 float columns (x, y, z) in normalised coords.
"""
import argparse
import csv
import json

from landmark_indices import (
    LEFT_IRIS, RIGHT_IRIS,
    LEFT_INNER, LEFT_OUTER, RIGHT_INNER, RIGHT_OUTER,
    LEFT_UPPER_EYELID, LEFT_LOWER_EYELID,
    RIGHT_UPPER_EYELID, RIGHT_LOWER_EYELID,
    LEFT_EYEBROW, RIGHT_EYEBROW,
    NUM_LANDMARKS,
)
from analytics import RealtimeAnalyzer

# Mean of iris landmarks for centre estimation
_LEFT_IRIS_CENTER  = LEFT_IRIS
_RIGHT_IRIS_CENTER = RIGHT_IRIS


def row_to_keypoints(row):
    """Convert a CSV row (label + 478*3 floats) to the keypoints dict used by analytics."""
    vals = row[1:1 + NUM_LANDMARKS * 3]
    try:
        vals = [float(x) if x != '' else 0.0 for x in vals]
    except (ValueError, TypeError):
        vals = [0.0] * (NUM_LANDMARKS * 3)

    pts = [(vals[i], vals[i + 1], vals[i + 2]) for i in range(0, len(vals), 3)]

    def safe(idx):
        if 0 <= idx < len(pts):
            return pts[idx]
        return (0.0, 0.0, 0.0)

    def iris_mean(indices):
        coords = [safe(i) for i in indices]
        n = len(coords)
        return tuple(sum(c[d] for c in coords) / n for d in range(3))

    return {
        'left_iris':           iris_mean(_LEFT_IRIS_CENTER),
        'right_iris':          iris_mean(_RIGHT_IRIS_CENTER),
        'left_inner':          safe(LEFT_INNER),
        'left_outer':          safe(LEFT_OUTER),
        'right_inner':         safe(RIGHT_INNER),
        'right_outer':         safe(RIGHT_OUTER),
        'left_eyebrow':        safe(LEFT_EYEBROW),       # canonical 105
        'right_eyebrow':       safe(RIGHT_EYEBROW),      # canonical 334
        'left_upper_eyelid':   safe(LEFT_UPPER_EYELID),
        'left_lower_eyelid':   safe(LEFT_LOWER_EYELID),
        'right_upper_eyelid':  safe(RIGHT_UPPER_EYELID),
        'right_lower_eyelid':  safe(RIGHT_LOWER_EYELID),
    }


def main():
    p = argparse.ArgumentParser(
        description='Run Sentin-Edge analytics on a preprocessed landmark CSV.')
    p.add_argument('--input_csv',    required=True, help='Path to landmark CSV')
    p.add_argument('--output_jsonl', required=True, help='Path to write JSONL results')
    args = p.parse_args()

    analyzer = RealtimeAnalyzer(window=12, calib_frames=10)

    with (open(args.input_csv, newline='', encoding='utf-8') as fh_in,
          open(args.output_jsonl, 'w', encoding='utf-8') as fh_out):
        reader = csv.reader(fh_in)
        for row in reader:
            if not row:
                continue
            label     = row[0]
            keypoints = row_to_keypoints(row)
            result    = analyzer.update(keypoints)

            out = {
                'label':        label,
                'gaze':         result['gaze_label'],
                'gaze_vals':    result['gaze_vals'],
                'stress_level': result['stress_level'],
                'stress_score': result['stress_score'],
                'integrity':    result['integrity'],
            }
            fh_out.write(json.dumps(out) + '\n')

    print('Done. Wrote results to', args.output_jsonl)


if __name__ == '__main__':
    main()
