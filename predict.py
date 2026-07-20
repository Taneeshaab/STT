import argparse
import csv
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import HistGradientBoostingClassifier

def extract_features_from_audio(wav_path, pause_start):
    if pause_start <= 0.05:
        return np.zeros(10)

    try:
        # Strict Causality: load audio ONLY up to pause_start
        y, sr = librosa.load(wav_path, sr=16000, duration=pause_start)
    except Exception:
        return np.zeros(10)

    if len(y) == 0:
        return np.zeros(10)

    window_samples = int(1.5 * sr)
    y_window = y[-window_samples:] if len(y) > window_samples else y

    rms = librosa.feature.rms(y=y_window)[0]
    mean_rms = np.mean(rms) if len(rms) > 0 else 0.0
    max_rms = np.max(rms) if len(rms) > 0 else 0.0

    last_200ms_samples = int(0.2 * sr)
    rms_last = librosa.feature.rms(y=y[-last_200ms_samples:])[0] if len(y) >= last_200ms_samples else rms
    mean_rms_last = np.mean(rms_last) if len(rms_last) > 0 else 0.0
    energy_drop_ratio = mean_rms_last / (mean_rms + 1e-6)

    spec_flat = np.mean(librosa.feature.spectral_flatness(y=y_window))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y_window, sr=sr))

    try:
        f0 = librosa.yin(y_window, fmin=60, fmax=300)
        valid_f0 = f0[~np.isnan(f0)]
        mean_f0 = np.mean(valid_f0) if len(valid_f0) > 0 else 0.0
        std_f0 = np.std(valid_f0) if len(valid_f0) > 0 else 0.0
        f0_slope = (valid_f0[-1] - valid_f0[0]) if len(valid_f0) > 1 else 0.0
    except Exception:
        mean_f0, std_f0, f0_slope = 0.0, 0.0, 0.0

    speech_duration = len(y) / sr

    return np.array([
        mean_rms, max_rms, mean_rms_last, energy_drop_ratio,
        spec_flat, spec_cent, mean_f0, std_f0, f0_slope, speech_duration
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    labels_path = os.path.join(args.data_dir, "labels.csv")
    df = pd.read_csv(labels_path)

    print(f"Processing {len(df)} audio samples...")
    X, y_train = [], []
    has_labels = "label" in df.columns

    for _, row in df.iterrows():
        wav_path = os.path.join(args.data_dir, row["audio_file"])
        feats = extract_features_from_audio(wav_path, float(row["pause_start"]))
        X.append(feats)
        if has_labels:
            y_train.append(1 if row["label"] == "eot" else 0)

    X = np.array(X)

    if has_labels:
        print("Training model on CPU...")
        model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
        model.fit(X, y_train)
        p_eot = model.predict_proba(X)[:, 1]
    else:
        p_eot = np.ones(len(df))

    out_rows = []
    for idx, row in df.iterrows():
        out_rows.append({
            "turn_id": row["turn_id"],
            "pause_index": row["pause_index"],
            "p_eot": round(float(p_eot[idx]), 4)
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"Saved predictions to {args.out}")

if __name__ == "__main__":
    main()