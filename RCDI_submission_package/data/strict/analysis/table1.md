| Policy | Mean cost (L-answer units) [95% CI] | Uncaught errors (% of queries) [95% CI] | Sent to human (% of queries) [95% CI] |
|---|---|---|---|
| Always send to a human | 5.00 [5.00, 5.00] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] |
| Always accept L (32B) | 6.99 [6.56, 7.41] | 24.0 [22.2, 25.7] | 0.0 [0.0, 0.0] |
| Always accept M (7B) | 9.84 [9.36, 10.35] | 38.5 [36.6, 40.5] | 0.0 [0.0, 0.0] |
| Always accept S (1.5B) | 13.05 [12.55, 13.54] | 52.0 [50.0, 54.0] | 0.0 [0.0, 0.0] |
| Cascade, tuned thresholds, raw log-prob | 4.98 [4.92, 5.04] | 0.9 [0.7, 1.2] | 93.4 [92.7, 94.1] |
| Cascade, expected-cost rule, raw log-prob | 10.22 [9.75, 10.73] | 39.1 [37.2, 41.1] | 2.2 [1.7, 2.8] |
| Cascade, expected-cost rule, Platt log-prob | 5.05 [5.03, 5.07] | 0.2 [0.1, 0.3] | 99.1 [98.9, 99.3] |
| Cascade, expected-cost rule, isotonic log-prob | 5.09 [5.03, 5.14] | 0.8 [0.6, 1.1] | 96.6 [96.1, 97.1] |
| RCDI: multi-signal risk model + expected-cost rule | 4.84 [4.72, 4.95] | 1.8 [1.4, 2.3] | 82.9 [81.5, 84.3] |
| RCDI + knowledge rule R2 | 4.84 [4.72, 4.95] | 1.8 [1.4, 2.3] | 82.9 [81.5, 84.3] |
| RCDI + conformal risk control, error cost raised (alpha = 0.05) | 4.84 [4.72, 4.95] | 1.8 [1.4, 2.3] | 83.1 [81.7, 84.5] |
| Cascade, common threshold + conformal risk control (alpha = 0.05) | 7.53 [7.29, 7.78] | 4.6 [3.9, 5.4] | 51.2 [49.5, 52.9] |
| L alone + conformal risk control (alpha = 0.05) | 7.87 [7.67, 8.07] | 3.6 [3.0, 4.2] | 51.2 [49.4, 53.0] |
