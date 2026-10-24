| Policy | Mean cost (L-answer units) [95% CI] | Uncaught errors (% of queries) [95% CI] | Sent to human (% of queries) [95% CI] |
|---|---|---|---|
| Always send to a human | 5.00 [5.00, 5.00] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] |
| Always accept L (32B) | 6.63 [6.21, 7.05] | 22.5 [20.8, 24.2] | 0.0 [0.0, 0.0] |
| Always accept M (7B) | 7.46 [7.02, 7.93] | 29.0 [27.2, 30.8] | 0.0 [0.0, 0.0] |
| Always accept S (1.5B) | 11.72 [11.21, 12.22] | 46.7 [44.7, 48.7] | 0.0 [0.0, 0.0] |
| Cascade, tuned thresholds, raw log-prob | 4.98 [4.92, 5.04] | 1.0 [0.8, 1.3] | 92.9 [92.2, 93.6] |
| Cascade, expected-cost rule, raw log-prob | 7.79 [7.35, 8.24] | 29.3 [27.6, 31.2] | 2.2 [1.7, 2.8] |
| Cascade, expected-cost rule, Platt log-prob | 5.06 [5.03, 5.09] | 0.4 [0.3, 0.6] | 98.2 [97.9, 98.5] |
| Cascade, expected-cost rule, isotonic log-prob | 5.09 [5.03, 5.14] | 0.8 [0.6, 1.1] | 96.6 [96.1, 97.1] |
| RCDI: multi-signal risk model + expected-cost rule | 4.93 [4.78, 5.10] | 2.9 [2.3, 3.5] | 67.5 [65.7, 69.3] |
| RCDI + knowledge rule R2 | 4.93 [4.78, 5.10] | 2.9 [2.3, 3.5] | 67.5 [65.7, 69.3] |
| RCDI + conformal risk control, error cost raised (alpha = 0.05) | 4.93 [4.78, 5.10] | 2.9 [2.3, 3.5] | 67.5 [65.8, 69.3] |
| Cascade, common threshold + conformal risk control (alpha = 0.05) | 7.07 [6.83, 7.32] | 4.8 [4.0, 5.5] | 50.6 [48.9, 52.4] |
| L alone + conformal risk control (alpha = 0.05) | 7.78 [7.58, 7.98] | 3.6 [3.0, 4.2] | 49.1 [47.4, 50.9] |
