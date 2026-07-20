# Run Log

- **Run 1**: Baseline silence agent.
  - *Score*: 1600 ms (English)
  - *Change/Reason*: Tested pure silence length gating as a reference anchor.

- **Run 2**: Baseline Logistic regression mapping simple raw energy.
  - *Score*: 1190 ms (English)
  - *Change/Reason*: Generated using the starter script to log early parameter boundaries.

- **Run 3**: Logistic Regression on high-resolution F0/Energy primitives.
  - *Score*: 1163 ms (English), 790 ms (Hindi)
  - *Change/Reason*: Brought in heavy multi-scale slope analysis on energy buffers and trailing voiced lengths via NumPy polynomials. The linear restriction failed to fully capture covariance mappings efficiently for English, but significantly improved Hindi performance structure.

- **Run 4**: Random Forest Estimator.
  - *Score*: 700 ms (English), 621 ms (Hindi)
  - *Change/Reason*: Migrated to an ensemble method with rigid structural constraints (`min_samples_leaf=4`, `max_depth=5`) to optimize decision boundaries and feature interaction processing. Latencies dropped significantly.
