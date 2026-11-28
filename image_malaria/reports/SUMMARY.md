# Summary - CKKS private inference on images (malaria)

## Data: malaria_28

- 27558 images, 201 patient/slide groups, positive rate 0.500, resized to 28x28 grayscale
- patient-level split (no patient in two splits): {'train': 19324, 'val': 4155, 'test': 4079}, positives {'train': 0.497, 'val': 0.488, 'test': 0.526}

## Plaintext models (test split)

| model | val AUROC | test AUROC | test accuracy |
|---|---|---|---|
| LR (pixels) | 0.7145 | 0.8066 | 0.6764 |
| CNN (4 ch, 7x7, stride 3, 457 params) | 0.8178 | 0.8568 | 0.7796 |

## Encrypted inference (step 12)

| model | poly | preset | encrypted on | AUROC plain | AUROC enc | acc plain | acc enc | mean err | max err | signed err | flips | flips (poly-calibrated thr) | s/patient |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn | 0 | precise | all (n=27558) | 0.8318 | 0.8318 | 0.7532 | 0.7532 | 2.13e-05 | 2.15e-04 | -3.4e-06 | 0 | 0 | 0.079 |
| cnn | 0 | standard | all (n=27558) | 0.8318 | 0.8318 | 0.7532 | 0.7534 | 4.44e-03 | 1.80e-02 | +2.1e-03 | 33 | 33 | 0.022 |
| lr | 3 | standard | all (n=27558) | 0.7361 | 0.7361 | 0.6316 | 0.6023 | 3.17e-02 | 5.71e-02 | -8.3e-04 | 1856 | 39 | 0.014 |
| lr | 5 | precise | all (n=27558) | 0.7361 | 0.7361 | 0.6316 | 0.6197 | 1.27e-02 | 2.07e-02 | +1.5e-04 | 770 | 0 | 0.049 |
| lr | 0 | precise | all (n=27558) | 0.7361 | 0.7361 | 0.6316 | 0.6316 | 9.15e-06 | 2.32e-05 | -7.8e-06 | 0 | 0 | 0.042 |
| lr | 0 | standard | all (n=27558) | 0.7361 | 0.7361 | 0.6316 | 0.6315 | 2.65e-03 | 6.16e-03 | +3.6e-05 | 19 | 19 | 0.013 |

Test-split rows:

| model | poly | preset | n | AUROC plain | AUROC enc | acc plain | acc enc | flips |
|---|---|---|---|---|---|---|---|---|
| cnn | 0 | precise | 4079 | 0.8568 | 0.8568 | 0.7796 | 0.7796 | 0 |
| cnn | 0 | standard | 4079 | 0.8568 | 0.8568 | 0.7796 | 0.7791 | 2 |
| lr | 3 | standard | 4079 | 0.8066 | 0.8066 | 0.6764 | 0.6426 | 240 |
| lr | 5 | precise | 4079 | 0.8066 | 0.8066 | 0.6764 | 0.6627 | 96 |
| lr | 0 | precise | 4079 | 0.8066 | 0.8066 | 0.6764 | 0.6764 | 0 |
| lr | 0 | standard | 4079 | 0.8066 | 0.8066 | 0.6764 | 0.6771 | 3 |

# Novelty 2 - SIMD batching (step 15)

| model | poly | preset | decision (scale pct) | patients | batch | wall s/patient | CPU ms/patient (enc + eval) | upload MB/patient | AUROC enc | flips |
|---|---|---|---|---|---|---|---|---|---|---|
| cnn | 0 | deep | False | 27558 | 8192 | 0.00773 | 15.14 | 0.1875 | 0.8318 | 0 |
| cnn | 0 | deep | True (100) | 27558 | 8192 | 0.00785 | 15.27 | 0.1875 | - | 0 |
| cnn | 0 | deep | True (95) | 27558 | 8192 | 0.00772 | 15.08 | 0.1875 | - | 115 |
| cnn | 0 | deep | True (99) | 27558 | 8192 | 0.00804 | 15.54 | 0.1875 | - | 30 |
| cnn | 0 | standard | False | 27558 | 4096 | 0.00337 | 5.82 | 0.0676 | 0.8318 | 0 |
| lr | 3 | standard | False | 27558 | 4096 | 0.00102 | 1.46 | 0.0677 | 0.7361 | 1877 |
| lr | 0 | precise | False | 27558 | 8192 | 0.00125 | 2.10 | 0.1199 | 0.7361 | 0 |
| lr | 0 | standard | False | 27558 | 4096 | 0.00101 | 1.46 | 0.0677 | 0.7361 | 2 |

# Novelty 1 - encrypted threshold decision (step 14 and 15 --decision)

| model | method | preset | scale pct | sign iters | patients | class flips | acc plain | acc enc | saturated (>=0.9) | Pearson(out, logit) | overflow (|x|>1.7) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn | per patient (14) | deep | 100.0 | 2 | 4079 | 0 | 0.7796 | 0.7796 | 0.001 | 0.991 | 0 |
| lr | per patient (14) | deep | 100.0 | 2 | 4079 | 0 | 0.6764 | 0.6764 | 0.009 | 0.989 | 0 |
| cnn | batched (15) | deep | 100.0 | 2 | 27558 | 0 | 0.7532 | 0.7532 | 0.001 | 0.993 | 0 |
| cnn | batched (15) | deep | 95.0 | 2 | 27558 | 115 | 0.7532 | 0.7497 | 0.196 | 0.165 | 281 |
| cnn | batched (15) | deep | 99.0 | 2 | 27558 | 30 | 0.7532 | 0.7525 | 0.047 | 0.161 | 54 |

`saturated` = share of patients whose decrypted value is within 0.1 of +-1 (the client learns only the class); the others are correctly signed but their magnitude still hints at the distance from the threshold. The score is hidden only for saturated patients (the others can be inverted to a score, the polynomial is monotone), so `saturated` is the leakage indicator. Pearson is dominated by a few huge logits and changes a lot with the scale choice alone - do not use it as a leakage measure.
