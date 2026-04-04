"""
benchmark_kalman.py

Compares baseline vs Kalman-smoothed pipelines on the same landmark CSV.
Produces a printed report with M1–M6 metrics for your project evaluation.

Usage:
  python benchmark_kalman.py --input_csv mpiigaze_landmarks.csv
"""
import argparse
import csv
import json
import time
import numpy as np
from collections import deque

from analytics import RealtimeAnalyzer, compute_gaze
from landmark_smoother import KeypointSmoother
from infer_from_landmarks import row_to_keypoints


def _iris_coords(kp):
    li = np.asarray(kp.get('left_iris', [0, 0]), dtype=np.float32)
    ri = np.asarray(kp.get('right_iris', [0, 0]), dtype=np.float32)
    return li, ri


def run_pipeline(rows, use_kalman: bool):
    analyzer = RealtimeAnalyzer(window=12, calib_frames=10)
    smoother = KeypointSmoother() if use_kalman else None

    iris_positions = []       # list of (lx, ly, rx, ry) per frame
    gaze_labels = []
    stress_scores = []
    integrity_scores = []
    frame_times = []
    prev_gaze = None
    gaze_flips = 0

    for row in rows:
        kp = row_to_keypoints(row)

        t0 = time.perf_counter()
        if smoother:
            kp = smoother.smooth(kp)
        result = analyzer.update(kp)
        frame_times.append((time.perf_counter() - t0) * 1000)

        li, ri = _iris_coords(kp)
        iris_positions.append((li[0], li[1], ri[0], ri[1]))

        gl = result['gaze_label']
        gaze_labels.append(gl)
        if prev_gaze is not None and gl != prev_gaze and gl != 'Calibrating':
            gaze_flips += 1
        prev_gaze = gl

        stress_scores.append(result['stress_score'])
        integrity_scores.append(result['integrity'])

    iris_arr = np.array(iris_positions)

    # M1 — jitter: mean std dev of iris x,y across consecutive frame differences
    diffs = np.abs(np.diff(iris_arr, axis=0))
    jitter = float(np.mean(np.std(diffs, axis=0)))

    # M2 — gaze flip rate (flips per 100 frames)
    flip_rate = gaze_flips / max(len(gaze_labels), 1) * 100

    # M3 — gaze zone distribution (proxy for stability: % Center frames)
    center_pct = gaze_labels.count('Center') / max(len(gaze_labels), 1) * 100

    # M4 — stress false positive proxy: frames where stress=High but gaze=Center
    fp_stress = sum(
        1 for g, s in zip(gaze_labels, stress_scores)
        if g == 'Center' and s >= 0.7
    ) / max(len(stress_scores), 1) * 100

    # M5 — CPU per frame
    cpu_ms = float(np.mean(frame_times))

    # M6 — integrity variance
    int_var = float(np.var(integrity_scores))

    return {
        'jitter_px': round(jitter, 3),
        'flip_rate_per100': round(flip_rate, 2),
        'center_pct': round(center_pct, 1),
        'stress_fp_pct': round(fp_stress, 2),
        'cpu_ms': round(cpu_ms, 3),
        'integrity_var': round(int_var, 3),
    }


def print_report(baseline, kalman):
    metrics = [
        ('M1  Iris jitter (px std)',      'jitter_px',        'lower'),
        ('M2  Gaze flip rate (/100f)',    'flip_rate_per100', 'lower'),
        ('M3  Center gaze %',             'center_pct',       'higher'),
        ('M4  Stress false positive %',   'stress_fp_pct',    'lower'),
        ('M5  CPU per frame (ms)',         'cpu_ms',           'lower'),
        ('M6  Integrity variance',         'integrity_var',    'lower'),
    ]
    print('\n' + '='*62)
    print(f"{'Metric':<30} {'Baseline':>10} {'Kalman':>10} {'Winner':>8}")
    print('-'*62)
    for label, key, direction in metrics:
        b, k = baseline[key], kalman[key]
        if direction == 'lower':
            winner = 'Kalman' if k < b else 'Baseline'
            delta = f"{((k - b) / max(abs(b), 1e-9)) * 100:+.1f}%"
        else:
            winner = 'Kalman' if k > b else 'Baseline'
            delta = f"{((k - b) / max(abs(b), 1e-9)) * 100:+.1f}%"
        print(f"{label:<30} {b:>10} {k:>10} {winner:>8}  {delta}")
    print('='*62 + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_csv', required=True)
    args = p.parse_args()

    rows = []
    with open(args.input_csv, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        for row in reader:
            if row:
                rows.append(row)

    print(f"Loaded {len(rows)} rows. Running baseline pipeline...")
    baseline = run_pipeline(rows, use_kalman=False)

    print("Running Kalman-smoothed pipeline...")
    kalman = run_pipeline(rows, use_kalman=True)

    print_report(baseline, kalman)


if __name__ == '__main__':
    main()