# Summary - CKKS private inference with hybrid CNN (time-series grid + static LR branch)

## All runs

| dataset | grid | target | stays | pos.rate | static-LR AUROC | CNN AUROC | preset | encrypted on | enc AUROC | max logit err | flips | s/patient |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic3 | 12x10 | los7 | 22536 | 0.251 | 0.6037 | 0.6924 | precise | all (n=22536) | 0.7003 | 1.98e-05 | 1 | 0.031 |
| mimic3 | 12x10 | los7 | 22536 | 0.251 | 0.6037 | 0.6924 | standard | all (n=22536) | 0.7003 | 6.38e-03 | 40 | 0.010 |
| mimic3 | 12x10 | los7 | 22536 | 0.251 | 0.6037 | 0.6924 | precise | all (n=22536) | 0.7210 | 8.49e-05 | 1 | 0.136 |
| mimic3 | 12x10 | los7 | 22536 | 0.251 | 0.6037 | 0.6924 | standard | all (n=22536) | 0.7210 | 1.19e-02 | 52 | 0.035 |
| mimic3 | 12x5 | los7 | 22536 | 0.251 | 0.6037 | 0.6417 | precise | all (n=22536) | 0.6467 | 1.72e-05 | 0 | 0.030 |
| mimic3 | 12x5 | los7 | 22536 | 0.251 | 0.6037 | 0.6417 | standard | all (n=22536) | 0.6466 | 2.74e-03 | 21 | 0.010 |
| mimic3 | 12x5 | los7 | 22536 | 0.251 | 0.6037 | 0.6417 | precise | all (n=22536) | 0.6609 | 9.06e-05 | 1 | 0.116 |
| mimic3 | 12x5 | los7 | 22536 | 0.251 | 0.6037 | 0.6417 | standard | all (n=22536) | 0.6609 | 1.15e-02 | 82 | 0.031 |
| mimic4 | 12x10 | los7 | 35794 | 0.232 | 0.6569 | 0.7852 | precise | all (n=35794) | 0.7698 | 3.16e-05 | 1 | 0.032 |
| mimic4 | 12x10 | los7 | 35794 | 0.232 | 0.6569 | 0.7852 | standard | all (n=35794) | 0.7698 | 6.02e-03 | 59 | 0.014 |
| mimic4 | 12x10 | los7 | 35794 | 0.232 | 0.6569 | 0.7852 | precise | all (n=35794) | 0.7815 | 8.64e-05 | 1 | 0.152 |
| mimic4 | 12x10 | los7 | 35794 | 0.232 | 0.6569 | 0.7852 | standard | all (n=35794) | 0.7815 | 1.57e-02 | 47 | 0.049 |
| mimic4 | 12x5 | los7 | 35794 | 0.232 | 0.6569 | 0.7049 | precise | all (n=35794) | 0.6860 | 1.91e-05 | 0 | 0.039 |
| mimic4 | 12x5 | los7 | 35794 | 0.232 | 0.6569 | 0.7049 | standard | all (n=35794) | 0.6860 | 8.00e-03 | 93 | 0.013 |
| mimic4 | 12x5 | los7 | 35794 | 0.232 | 0.6569 | 0.7049 | precise | all (n=35794) | 0.7096 | 1.06e-04 | 0 | 0.131 |
| mimic4 | 12x5 | los7 | 35794 | 0.232 | 0.6569 | 0.7049 | standard | all (n=35794) | 0.7095 | 1.79e-02 | 143 | 0.040 |

Static-LR and CNN AUROC columns are on the held-out test split (patient-level split). `encrypted on` = which patients were encrypted; `all` is the whole cohort (train + val + test), so its AUROC is optimistic - see the per-split lines below for test-only numbers. `flips` = encrypted prediction differs from plaintext prediction at the val-tuned threshold.

## mimic3_12x10_los7

- cohort: 22536 stays, positive rate 0.251, split {'train': 15776, 'val': 3380, 'test': 3380}
- vitals in grid (10): heart_rate, resp_rate, spo2, sbp, temp_c, dbp, mbp, gcs_motor, fio2, glucose
- stays with at least one observation: heart_rate 84%, resp_rate 84%, spo2 84%, sbp 39%, temp_c 83%, dbp 39%, mbp 84%, gcs_motor 44%, fio2 57%, glucose 88%
- static-only LR: test AUROC 0.6037
- CNN: val AUROC 0.7047, test AUROC 0.6924, test AUPRC 0.4463, threshold 0.43
- CKKS precise on all: n=22536, enc AUROC 0.7003, max logit err 1.98e-05, flips 1, 0.03 s/patient (wall, parallel)
    - test: n=3380, plain AUROC 0.6853, enc AUROC 0.6853, flips 0, acc plain/enc 0.6885/0.6885
    - train: n=15776, plain AUROC 0.7062, enc AUROC 0.7062, flips 1, acc plain/enc 0.7013/0.7013
    - val: n=3380, plain AUROC 0.6869, enc AUROC 0.6869, flips 0, acc plain/enc 0.6991/0.6991
- CKKS standard on all: n=22536, enc AUROC 0.7003, max logit err 6.38e-03, flips 40, 0.01 s/patient (wall, parallel)
    - test: n=3380, plain AUROC 0.6853, enc AUROC 0.6854, flips 6, acc plain/enc 0.6885/0.6879
    - train: n=15776, plain AUROC 0.7062, enc AUROC 0.7063, flips 25, acc plain/enc 0.7013/0.7013
    - val: n=3380, plain AUROC 0.6869, enc AUROC 0.6869, flips 9, acc plain/enc 0.6991/0.6988
- CKKS precise on all: n=22536, enc AUROC 0.7210, max logit err 8.49e-05, flips 1, 0.14 s/patient (wall, parallel)
    - test: n=3380, plain AUROC 0.6924, enc AUROC 0.6924, flips 0, acc plain/enc 0.5751/0.5751
    - train: n=15776, plain AUROC 0.7305, enc AUROC 0.7305, flips 1, acc plain/enc 0.5919/0.5920
    - val: n=3380, plain AUROC 0.7047, enc AUROC 0.7047, flips 0, acc plain/enc 0.5802/0.5802
- CKKS standard on all: n=22536, enc AUROC 0.7210, max logit err 1.19e-02, flips 52, 0.03 s/patient (wall, parallel)
    - test: n=3380, plain AUROC 0.6924, enc AUROC 0.6923, flips 8, acc plain/enc 0.5751/0.5769
    - train: n=15776, plain AUROC 0.7305, enc AUROC 0.7305, flips 34, acc plain/enc 0.5919/0.5923
    - val: n=3380, plain AUROC 0.7047, enc AUROC 0.7048, flips 10, acc plain/enc 0.5802/0.5802

## mimic3_12x5_los7

- cohort: 22536 stays, positive rate 0.251, split {'train': 15776, 'val': 3380, 'test': 3380}
- vitals in grid (5): heart_rate, resp_rate, spo2, sbp, temp_c
- stays with at least one observation: heart_rate 84%, resp_rate 84%, spo2 84%, sbp 39%, temp_c 83%, dbp 39%, mbp 84%, gcs_motor 44%, fio2 57%, glucose 88%
- static-only LR: test AUROC 0.6037
- CNN: val AUROC 0.6497, test AUROC 0.6417, test AUPRC 0.3997, threshold 0.47
- CKKS precise on all: n=22536, enc AUROC 0.6467, max logit err 1.72e-05, flips 0, 0.03 s/patient (wall, parallel)
    - test: n=3380, plain AUROC 0.6370, enc AUROC 0.6370, flips 0, acc plain/enc 0.5766/0.5766
    - train: n=15776, plain AUROC 0.6513, enc AUROC 0.6513, flips 0, acc plain/enc 0.5852/0.5852
    - val: n=3380, plain AUROC 0.6340, enc AUROC 0.6340, flips 0, acc plain/enc 0.5734/0.5734
- CKKS standard on all: n=22536, enc AUROC 0.6466, max logit err 2.74e-03, flips 21, 0.01 s/patient (wall, parallel)
    - test: n=3380, plain AUROC 0.6370, enc AUROC 0.6370, flips 6, acc plain/enc 0.5766/0.5766
    - train: n=15776, plain AUROC 0.6513, enc AUROC 0.6513, flips 11, acc plain/enc 0.5852/0.5854
    - val: n=3380, plain AUROC 0.6340, enc AUROC 0.6339, flips 4, acc plain/enc 0.5734/0.5734
- CKKS precise on all: n=22536, enc AUROC 0.6609, max logit err 9.06e-05, flips 1, 0.12 s/patient (wall, parallel)
    - test: n=3380, plain AUROC 0.6417, enc AUROC 0.6417, flips 0, acc plain/enc 0.5769/0.5769
    - train: n=15776, plain AUROC 0.6674, enc AUROC 0.6674, flips 1, acc plain/enc 0.5913/0.5914
    - val: n=3380, plain AUROC 0.6497, enc AUROC 0.6497, flips 0, acc plain/enc 0.5985/0.5985
- CKKS standard on all: n=22536, enc AUROC 0.6609, max logit err 1.15e-02, flips 82, 0.03 s/patient (wall, parallel)
    - test: n=3380, plain AUROC 0.6417, enc AUROC 0.6417, flips 14, acc plain/enc 0.5769/0.5763
    - train: n=15776, plain AUROC 0.6674, enc AUROC 0.6673, flips 57, acc plain/enc 0.5913/0.5922
    - val: n=3380, plain AUROC 0.6497, enc AUROC 0.6498, flips 11, acc plain/enc 0.5985/0.5982

## mimic4_12x10_los7

- cohort: 35794 stays, positive rate 0.232, split {'train': 25056, 'val': 5369, 'test': 5369}
- vitals in grid (10): heart_rate, resp_rate, spo2, sbp, temp_c, dbp, mbp, gcs_motor, fio2, glucose
- stays with at least one observation: heart_rate 100%, resp_rate 100%, spo2 100%, sbp 100%, temp_c 98%, dbp 100%, mbp 100%, gcs_motor 100%, fio2 63%, glucose 100%
- static-only LR: test AUROC 0.6569
- CNN: val AUROC 0.7772, test AUROC 0.7852, test AUPRC 0.5106, threshold 0.51
- CKKS precise on all: n=35794, enc AUROC 0.7698, max logit err 3.16e-05, flips 1, 0.03 s/patient (wall, parallel)
    - test: n=5369, plain AUROC 0.7741, enc AUROC 0.7741, flips 1, acc plain/enc 0.7160/0.7158
    - train: n=25056, plain AUROC 0.7703, enc AUROC 0.7703, flips 0, acc plain/enc 0.7160/0.7160
    - val: n=5369, plain AUROC 0.7629, enc AUROC 0.7629, flips 0, acc plain/enc 0.7059/0.7059
- CKKS standard on all: n=35794, enc AUROC 0.7698, max logit err 6.02e-03, flips 59, 0.01 s/patient (wall, parallel)
    - test: n=5369, plain AUROC 0.7741, enc AUROC 0.7740, flips 12, acc plain/enc 0.7160/0.7148
    - train: n=25056, plain AUROC 0.7703, enc AUROC 0.7703, flips 36, acc plain/enc 0.7160/0.7159
    - val: n=5369, plain AUROC 0.7629, enc AUROC 0.7629, flips 11, acc plain/enc 0.7059/0.7053
- CKKS precise on all: n=35794, enc AUROC 0.7815, max logit err 8.64e-05, flips 1, 0.15 s/patient (wall, parallel)
    - test: n=5369, plain AUROC 0.7852, enc AUROC 0.7852, flips 0, acc plain/enc 0.7499/0.7499
    - train: n=25056, plain AUROC 0.7817, enc AUROC 0.7817, flips 0, acc plain/enc 0.7470/0.7470
    - val: n=5369, plain AUROC 0.7772, enc AUROC 0.7772, flips 1, acc plain/enc 0.7415/0.7417
- CKKS standard on all: n=35794, enc AUROC 0.7815, max logit err 1.57e-02, flips 47, 0.05 s/patient (wall, parallel)
    - test: n=5369, plain AUROC 0.7852, enc AUROC 0.7852, flips 7, acc plain/enc 0.7499/0.7489
    - train: n=25056, plain AUROC 0.7817, enc AUROC 0.7817, flips 29, acc plain/enc 0.7470/0.7468
    - val: n=5369, plain AUROC 0.7772, enc AUROC 0.7770, flips 11, acc plain/enc 0.7415/0.7402

## mimic4_12x5_los7

- cohort: 35794 stays, positive rate 0.232, split {'train': 25056, 'val': 5369, 'test': 5369}
- vitals in grid (5): heart_rate, resp_rate, spo2, sbp, temp_c
- stays with at least one observation: heart_rate 100%, resp_rate 100%, spo2 100%, sbp 100%, temp_c 98%, dbp 100%, mbp 100%, gcs_motor 100%, fio2 63%, glucose 100%
- static-only LR: test AUROC 0.6569
- CNN: val AUROC 0.7042, test AUROC 0.7049, test AUPRC 0.4063, threshold 0.49
- CKKS precise on all: n=35794, enc AUROC 0.6860, max logit err 1.91e-05, flips 0, 0.04 s/patient (wall, parallel)
    - test: n=5369, plain AUROC 0.6874, enc AUROC 0.6874, flips 0, acc plain/enc 0.6681/0.6681
    - train: n=25056, plain AUROC 0.6868, enc AUROC 0.6868, flips 0, acc plain/enc 0.6720/0.6720
    - val: n=5369, plain AUROC 0.6816, enc AUROC 0.6816, flips 0, acc plain/enc 0.6696/0.6696
- CKKS standard on all: n=35794, enc AUROC 0.6860, max logit err 8.00e-03, flips 93, 0.01 s/patient (wall, parallel)
    - test: n=5369, plain AUROC 0.6874, enc AUROC 0.6874, flips 20, acc plain/enc 0.6681/0.6677
    - train: n=25056, plain AUROC 0.6868, enc AUROC 0.6868, flips 62, acc plain/enc 0.6720/0.6708
    - val: n=5369, plain AUROC 0.6816, enc AUROC 0.6816, flips 11, acc plain/enc 0.6696/0.6679
- CKKS precise on all: n=35794, enc AUROC 0.7096, max logit err 1.06e-04, flips 0, 0.13 s/patient (wall, parallel)
    - test: n=5369, plain AUROC 0.7049, enc AUROC 0.7049, flips 0, acc plain/enc 0.6331/0.6331
    - train: n=25056, plain AUROC 0.7118, enc AUROC 0.7118, flips 0, acc plain/enc 0.6390/0.6390
    - val: n=5369, plain AUROC 0.7042, enc AUROC 0.7042, flips 0, acc plain/enc 0.6420/0.6420
- CKKS standard on all: n=35794, enc AUROC 0.7095, max logit err 1.79e-02, flips 143, 0.04 s/patient (wall, parallel)
    - test: n=5369, plain AUROC 0.7049, enc AUROC 0.7047, flips 17, acc plain/enc 0.6331/0.6333
    - train: n=25056, plain AUROC 0.7118, enc AUROC 0.7117, flips 102, acc plain/enc 0.6390/0.6387
    - val: n=5369, plain AUROC 0.7042, enc AUROC 0.7041, flips 24, acc plain/enc 0.6420/0.6398
