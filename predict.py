import argparse
import csv
import os
import joblib

import numpy as np
import librosa

from features import load_wav, speech_before, frame_energy_db, f0_contour

def extract_features(x, sr, pause_start):
    seg = speech_before(x, sr, pause_start, window_s=2.5) # longer context
    if len(seg) < sr // 10:
        return np.zeros(20, dtype=np.float32)

    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    
    ctx_len = len(seg) / sr
    mean_energy = e.mean() if len(e) > 0 else 0.0
    
    voiced_mask = f0 > 0
    voiced = f0[voiced_mask]
    
    e_last = e[-5:].mean() if len(e) >= 5 else (e[-1] if len(e) > 0 else 0.0)
    e_decay = e_last - mean_energy
    
    last_voiced_f0 = 0.0
    f0_slope10 = 0.0
    f0_slope30 = 0.0
    f0_diff = 0.0
    last_voiced_duration = 0.0
    
    if len(voiced) > 0:
        last_voiced_f0 = voiced[-3:].mean() if len(voiced) >= 3 else voiced[-1]
        f0_diff = last_voiced_f0 - voiced.mean()
        
        def get_slope(v, N):
            v_N = v[-N:]
            if len(v_N) < 2: return 0.0
            x_idx = np.arange(len(v_N))
            slope = np.polyfit(x_idx, v_N, 1)[0]
            return slope
            
        f0_slope10 = get_slope(voiced, 10)
        f0_slope30 = get_slope(voiced, 30)

        idx = np.where(~voiced_mask)[0]
        if len(idx) == 0:
            last_voiced_duration = len(voiced_mask)
        else:
            last_voiced_duration = 0
            for i in range(len(voiced_mask)-1, -1, -1):
                if voiced_mask[i]:
                    last_voiced_duration += 1
                else:
                    break
    
    def get_e_slope(e_arr, N):
        e_N = e_arr[-N:]
        if len(e_N) < 2: return 0.0
        return np.polyfit(np.arange(len(e_N)), e_N, 1)[0]
        
    e_slope10 = get_e_slope(e, 10)
    e_slope30 = get_e_slope(e, 30)
    e_slope50 = get_e_slope(e, 50)
    
    last_200ms = seg[-int(sr*0.2):] if len(seg) > int(sr*0.2) else seg
    if len(last_200ms) > 0:
        zcr = librosa.feature.zero_crossing_rate(y=last_200ms)[0].mean()
    else:
        zcr = 0.0
        
    last_500m_frames = 50 
    if len(voiced_mask) >= last_500m_frames:
        voiced_ratio_500 = voiced_mask[-last_500m_frames:].mean()
    else:
        voiced_ratio_500 = voiced_mask.mean() if len(voiced_mask) > 0 else 0.0

    return np.array([
        ctx_len,
        mean_energy, e_last, e_decay, e_slope10, e_slope30, e_slope50,
        last_voiced_f0, f0_diff, f0_slope10, f0_slope30,
        last_voiced_duration,
        zcr,
        voiced_ratio_500,
        voiced.mean() if len(voiced) > 0 else 0.0,
        voiced.std() if len(voiced) > 0 else 0.0,
        e.std() if len(e) > 0 else 0.0,
        e[-10:].mean() if len(e) >= 10 else 0.0,
        get_e_slope(e, 5),
        get_slope(voiced, 5) if len(voiced) > 0 else 0.0
    ], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    # Determine script location to be able to load model.pkl relative to predict.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "ckpt.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ckpt.pt not found at {model_path}.")
    
    clf = joblib.load(model_path)
    
    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    
    path_cache = {}
    preds = []
    
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in path_cache:
            path_cache[path] = load_wav(path)
        
        x, sr = path_cache[path]
        feat = extract_features(x, sr, float(r["pause_start"]))
        
        # predict_proba returns shape (1, 2). Get positive class prob.
        prob = clf.predict_proba([feat])[0, 1]
        
        preds.append({
            "turn_id": r["turn_id"],
            "pause_index": r["pause_index"],
            "p_eot": prob
        })
        
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        w.writeheader()
        w.writerows([{"turn_id": p["turn_id"], "pause_index": p["pause_index"], "p_eot": f"{p['p_eot']:.4f}"} for p in preds])
        
    print(f"wrote {len(preds)} predictions -> {args.out}")

if __name__ == "__main__":
    main()
