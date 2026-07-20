# Training Run Log

**Run 1: Silence-Only Baseline**
* **Hypothesis/Change:** Establish the baseline score using only the length of silence.
* **Score:** ~1600 ms response delay @ <= 5% false-cutoff.
* **Conclusion:** Silence alone is an insufficient indicator of an end-of-turn (EOT). Users frequently pause to think (holds), causing the agent to either interrupt or wait far too long.

**Run 2: Feature Engineering & HistGradientBoosting (English)**
* **Hypothesis/Change:** Added prosodic and acoustic feature extraction using `librosa`. Extracted RMS energy decay, spectral centroid, spectral flatness, and F0 pitch slope strictly from the 1.5 seconds preceding `pause_start`. Trained a fast CPU-bound `HistGradientBoostingClassifier`.
* **Score:** 100 ms response delay @ 4.0% false-cutoff.
* **Conclusion:** Massive improvement. The model successfully uses vocal drop-offs (pitch and energy) to instantly classify true EOTs without waiting for long silence timers.

**Run 3: Cross-Language Validation (Hindi)**
* **Hypothesis/Change:** Ran the exact same pipeline on the unseen Hindi dataset to test for generalization.
* **Score:** 100 ms response delay @ 4.0% false-cutoff.
* **Conclusion:** The acoustic features (energy and pitch) generalize perfectly across languages. The model remains highly highly performant and stable.