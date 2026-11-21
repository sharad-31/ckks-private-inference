# Summary - CKKS private inference on images (malaria)

## Data: malaria_28

- 27558 images, 201 patient/slide groups, positive rate 0.500, resized to 28x28 grayscale
- patient-level split (no patient in two splits): {'train': 19324, 'val': 4155, 'test': 4079}, positives {'train': 0.497, 'val': 0.488, 'test': 0.526}

## Plaintext models (test split)

| model | val AUROC | test AUROC | test accuracy |
|---|---|---|---|
| LR (pixels) | 0.7146 | 0.8066 | 0.6764 |
| CNN (4 ch, 7x7, stride 3, 457 params) | 0.8178 | 0.8569 | 0.7779 |

## Encrypted inference (step 12)

| model | poly | preset | encrypted on | AUROC plain | AUROC enc | acc plain | acc enc | mean err | max err | signed err | flips | flips (poly-calibrated thr) | s/patient |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn | 0 | precise | all (n=27558) | 0.8319 | 0.8319 | 0.7518 | 0.7518 | 1.72e-05 | 2.27e-04 | +4.4e-07 | 0 | 0 | 0.105 |
| cnn | 0 | standard | all (n=27558) | 0.8319 | 0.8318 | 0.7518 | 0.7518 | 3.46e-03 | 1.15e-02 | -8.6e-04 | 29 | 29 | 0.032 |
| lr | 3 | standard | all (n=27558) | 0.7361 | 0.7361 | 0.6316 | 0.6023 | 3.16e-02 | 5.77e-02 | -6.3e-04 | 1858 | 46 | 0.015 |
| lr | 5 | precise | all (n=27558) | 0.7361 | 0.7361 | 0.6316 | 0.6196 | 1.26e-02 | 2.06e-02 | +1.4e-04 | 770 | 0 | 0.058 |
| lr | 0 | precise | all (n=27558) | 0.7361 | 0.7361 | 0.6316 | 0.6316 | 6.41e-06 | 1.80e-05 | +2.0e-06 | 0 | 0 | 0.047 |

Test-split rows:

| model | poly | preset | n | AUROC plain | AUROC enc | acc plain | acc enc | flips |
|---|---|---|---|---|---|---|---|---|
| cnn | 0 | precise | 4079 | 0.8569 | 0.8569 | 0.7779 | 0.7779 | 0 |
| cnn | 0 | standard | 4079 | 0.8569 | 0.8568 | 0.7779 | 0.7776 | 3 |
| lr | 3 | standard | 4079 | 0.8066 | 0.8066 | 0.6764 | 0.6440 | 236 |
| lr | 5 | precise | 4079 | 0.8066 | 0.8066 | 0.6764 | 0.6629 | 97 |
| lr | 0 | precise | 4079 | 0.8066 | 0.8066 | 0.6764 | 0.6764 | 0 |

# Novelty 2 - SIMD batching (step 15)

| model | poly | preset | decision (scale pct) | patients | batch | wall s/patient | CPU ms/patient (enc + eval) | upload MB/patient | AUROC enc | flips |
|---|---|---|---|---|---|---|---|---|---|---|
| cnn | 0 | deep | True (95) | 27558 | 8192 | 0.01015 | 19.88 | 0.1874 | - | 117 |
| cnn | 0 | deep | True (99) | 27558 | 8192 | 0.00978 | 19.14 | 0.1874 | - | 30 |
| cnn | 0 | standard | False | 27558 | 4096 | 0.00454 | 7.78 | 0.0677 | 0.8319 | 0 |
| lr | 3 | standard | False | 27558 | 4096 | 0.00136 | 2.01 | 0.0677 | 0.7361 | 1878 |

# Novelty 1 - encrypted threshold decision (step 14 and 15 --decision)

| model | method | preset | scale pct | sign iters | patients | class flips | acc plain | acc enc | saturated (>=0.9) | Pearson(out, logit) | overflow (|x|>1.7) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn | per patient (14) | deep | 100.0 | 2 | 4079 | 0 | 0.7779 | 0.7779 | 0.001 | 0.991 | 0 |
| lr | per patient (14) | deep | 100.0 | 2 | 4079 | 0 | 0.6764 | 0.6764 | 0.009 | 0.989 | 0 |
| cnn | batched (15) | deep | 95.0 | 2 | 27558 | 117 | 0.7518 | 0.7482 | 0.195 | 0.165 | 281 |
| cnn | batched (15) | deep | 99.0 | 2 | 27558 | 30 | 0.7518 | 0.7511 | 0.048 | 0.162 | 54 |

`saturated` = share of patients whose decrypted value is within 0.1 of +-1 (the client learns only the class); the others are correctly signed but their magnitude still hints at the distance from the threshold. The score is hidden only for saturated patients (the others can be inverted to a score, the polynomial is monotone), so `saturated` is the leakage indicator. Pearson is dominated by a few huge logits and changes a lot with the scale choice alone - do not use it as a leakage measure.
