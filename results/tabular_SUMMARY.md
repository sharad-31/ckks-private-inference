# Summary - CKKS private inference with hybrid CNN (time-series grid + static LR branch)

## All runs

| dataset | grid | target | stays | pos.rate | static-LR AUROC | CNN AUROC | preset | encrypted on | enc AUROC | max logit err | flips | s/patient |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mimic3 | 12x10 | los7 | 22536 | 0.251 | 0.6037 | 0.6924 | precise | all (n=22536) | 0.7210 | 6.25e-05 | 1 | 0.154 |
| mimic3 | 12x10 | los7 | 22536 | 0.251 | 0.6037 | 0.6924 | standard | all (n=22536) | 0.7210 | 1.77e-02 | 73 | 0.050 |
| mimic3 | 12x5 | los7 | 22536 | 0.251 | 0.6037 | 0.6417 | precise | all (n=22536) | 0.6609 | 5.96e-05 | 0 | 0.143 |
| mimic3 | 12x5 | los7 | 22536 | 0.251 | 0.6037 | 0.6417 | standard | all (n=22536) | 0.6609 | 1.56e-02 | 107 | 0.044 |
| mimic4 | 12x10 | los7 | 35794 | 0.232 | 0.6569 | 0.7852 | precise | all (n=35794) | 0.7815 | 8.48e-05 | 0 | 0.135 |
| mimic4 | 12x10 | los7 | 35794 | 0.232 | 0.6569 | 0.7852 | standard | all (n=35794) | 0.7815 | 1.36e-02 | 65 | 0.040 |
| mimic4 | 12x5 | los7 | 35794 | 0.232 | 0.6569 | 0.7049 | precise | all (n=35794) | 0.7096 | 8.20e-05 | 0 | 0.099 |
| mimic4 | 12x5 | los7 | 35794 | 0.232 | 0.6569 | 0.7049 | standard | all (n=35794) | 0.7096 | 9.11e-03 | 58 | 0.029 |

Static-LR and CNN AUROC columns are on the held-out test split (patient-level split). `encrypted on` = which patients were encrypted; `all` is the whole cohort (train + val + test), so its AUROC is optimistic - see the per-split lines below for test-only numbers. `flips` = encrypted prediction differs from plaintext prediction at the val-tuned threshold.

## mimic3_12x10_los7

- cohort: 22536 stays, positive rate 0.251, split {'train': 15776, 'val': 3380, 'test': 3380}
- vitals in grid (10): heart_rate, resp_rate, spo2, sbp, temp_c, dbp, mbp, gcs_motor, fio2, glucose
- stays with at least one observation: heart_rate 84%, resp_rate 84%, spo2 84%, sbp 39%, temp_c 83%, dbp 39%, mbp 84%, gcs_motor 44%, fio2 57%, glucose 88%
- static-only LR: test AUROC 0.6037
- CNN: val AUROC 0.7047, test AUROC 0.6924, test AUPRC 0.4463, threshold 0.43
- CKKS precise on all: n=22536, enc AUROC 0.7210, max logit err 6.25e-05, flips 1, 0.15 s/patient (wall, parallel)
    - train: n=15776, plain AUROC 0.7305, enc AUROC 0.7305, flips 1, acc plain/enc 0.5919/0.5920
    - val: n=3380, plain AUROC 0.7047, enc AUROC 0.7047, flips 0, acc plain/enc 0.5802/0.5802
    - test: n=3380, plain AUROC 0.6924, enc AUROC 0.6924, flips 0, acc plain/enc 0.5751/0.5751
- CKKS standard on all: n=22536, enc AUROC 0.7210, max logit err 1.77e-02, flips 73, 0.05 s/patient (wall, parallel)
    - train: n=15776, plain AUROC 0.7305, enc AUROC 0.7305, flips 54, acc plain/enc 0.5919/0.5928
    - val: n=3380, plain AUROC 0.7047, enc AUROC 0.7047, flips 11, acc plain/enc 0.5802/0.5799
    - test: n=3380, plain AUROC 0.6924, enc AUROC 0.6925, flips 8, acc plain/enc 0.5751/0.5763

## mimic3_12x5_los7

- cohort: 22536 stays, positive rate 0.251, split {'train': 15776, 'val': 3380, 'test': 3380}
- vitals in grid (5): heart_rate, resp_rate, spo2, sbp, temp_c
- stays with at least one observation: heart_rate 84%, resp_rate 84%, spo2 84%, sbp 39%, temp_c 83%, dbp 39%, mbp 84%, gcs_motor 44%, fio2 57%, glucose 88%
- static-only LR: test AUROC 0.6037
- CNN: val AUROC 0.6497, test AUROC 0.6417, test AUPRC 0.3997, threshold 0.47
- CKKS precise on all: n=22536, enc AUROC 0.6609, max logit err 5.96e-05, flips 0, 0.14 s/patient (wall, parallel)
    - train: n=15776, plain AUROC 0.6674, enc AUROC 0.6674, flips 0, acc plain/enc 0.5913/0.5913
    - val: n=3380, plain AUROC 0.6497, enc AUROC 0.6497, flips 0, acc plain/enc 0.5985/0.5985
    - test: n=3380, plain AUROC 0.6417, enc AUROC 0.6417, flips 0, acc plain/enc 0.5769/0.5769
- CKKS standard on all: n=22536, enc AUROC 0.6609, max logit err 1.56e-02, flips 107, 0.04 s/patient (wall, parallel)
    - train: n=15776, plain AUROC 0.6674, enc AUROC 0.6673, flips 82, acc plain/enc 0.5913/0.5903
    - val: n=3380, plain AUROC 0.6497, enc AUROC 0.6502, flips 17, acc plain/enc 0.5985/0.5988
    - test: n=3380, plain AUROC 0.6417, enc AUROC 0.6415, flips 8, acc plain/enc 0.5769/0.5757

## mimic4_12x10_los7

- cohort: 35794 stays, positive rate 0.232, split {'train': 25056, 'val': 5369, 'test': 5369}
- vitals in grid (10): heart_rate, resp_rate, spo2, sbp, temp_c, dbp, mbp, gcs_motor, fio2, glucose
- stays with at least one observation: heart_rate 100%, resp_rate 100%, spo2 100%, sbp 100%, temp_c 98%, dbp 100%, mbp 100%, gcs_motor 100%, fio2 63%, glucose 100%
- static-only LR: test AUROC 0.6569
- CNN: val AUROC 0.7772, test AUROC 0.7852, test AUPRC 0.5106, threshold 0.51
- CKKS precise on all: n=35794, enc AUROC 0.7815, max logit err 8.48e-05, flips 0, 0.13 s/patient (wall, parallel)
    - train: n=25056, plain AUROC 0.7817, enc AUROC 0.7817, flips 0, acc plain/enc 0.7470/0.7470
    - val: n=5369, plain AUROC 0.7772, enc AUROC 0.7772, flips 0, acc plain/enc 0.7415/0.7415
    - test: n=5369, plain AUROC 0.7852, enc AUROC 0.7852, flips 0, acc plain/enc 0.7499/0.7499
- CKKS standard on all: n=35794, enc AUROC 0.7815, max logit err 1.36e-02, flips 65, 0.04 s/patient (wall, parallel)
    - train: n=25056, plain AUROC 0.7817, enc AUROC 0.7817, flips 42, acc plain/enc 0.7470/0.7470
    - val: n=5369, plain AUROC 0.7772, enc AUROC 0.7772, flips 11, acc plain/enc 0.7415/0.7409
    - test: n=5369, plain AUROC 0.7852, enc AUROC 0.7852, flips 12, acc plain/enc 0.7499/0.7484

## mimic4_12x5_los7

- cohort: 35794 stays, positive rate 0.232, split {'train': 25056, 'val': 5369, 'test': 5369}
- vitals in grid (5): heart_rate, resp_rate, spo2, sbp, temp_c
- stays with at least one observation: heart_rate 100%, resp_rate 100%, spo2 100%, sbp 100%, temp_c 98%, dbp 100%, mbp 100%, gcs_motor 100%, fio2 63%, glucose 100%
- static-only LR: test AUROC 0.6569
- CNN: val AUROC 0.7042, test AUROC 0.7049, test AUPRC 0.4063, threshold 0.49
- CKKS precise on all: n=35794, enc AUROC 0.7096, max logit err 8.20e-05, flips 0, 0.10 s/patient (wall, parallel)
    - train: n=25056, plain AUROC 0.7118, enc AUROC 0.7118, flips 0, acc plain/enc 0.6390/0.6390
    - val: n=5369, plain AUROC 0.7042, enc AUROC 0.7042, flips 0, acc plain/enc 0.6420/0.6420
    - test: n=5369, plain AUROC 0.7049, enc AUROC 0.7049, flips 0, acc plain/enc 0.6331/0.6331
- CKKS standard on all: n=35794, enc AUROC 0.7096, max logit err 9.11e-03, flips 58, 0.03 s/patient (wall, parallel)
    - train: n=25056, plain AUROC 0.7118, enc AUROC 0.7118, flips 39, acc plain/enc 0.6390/0.6385
    - val: n=5369, plain AUROC 0.7042, enc AUROC 0.7042, flips 12, acc plain/enc 0.6420/0.6416
    - test: n=5369, plain AUROC 0.7049, enc AUROC 0.7048, flips 7, acc plain/enc 0.6331/0.6329
