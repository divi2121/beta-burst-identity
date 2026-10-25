# Results

All results are on Zhou2016 (4 people, 3 sessions on different days, 9 central channels). Unless stated otherwise they are cross-day: sessions 0 and 1 train, session 2 tests. Code for each finding is listed in the README.

## 1. Burst waveform shape identifies people

Classifiers on burst shape features reach 0.50 (mean component scores) and 0.56 (shape-type histogram) balanced accuracy across days, against 0.25 chance (permutation p ≤ 0.001, 1000 permutations). Within a session: 0.51 and 0.49. A simpler Hilbert-based detector gives lower but above-chance scores.

## 2. Artifact rejection keeps that information, and helps the covariance fingerprint

I compared a global peak-to-peak threshold, AutoReject and FAAR, each against removing the same number of random epochs (20 draws).
- Across days, all three make a Riemannian covariance fingerprint more confident than random removal does: log-loss 0.036 with no rejection, 0.021 to 0.024 after rejection, below every random range.
- Within a session, only FAAR is below its random range (0.008 vs 0.010 to 0.017).
- No method reduces burst shape identity, and no method preferentially removes burst-rich epochs.

## 3. FAAR rejection rates

This is my implementation of section II.A of the paper, with my own choices where it does not specify one. It is not the authors' code, and I did not use their settings. It rejects 13 to 45% of epochs per session, against 2.3% or less reported in the paper. The sorted quality curve in my implementation has no clear knee, so the knee detection cuts deep. This may reflect my choices and not the method, so the FAAR results in section 2 describe this implementation only. It also picks different epochs from AutoReject.

## 4. A burst-derived fingerprint is a spectral effect, and a standard filter bank does better

I built filters from the burst waveforms (PCA on bursts, axes chosen by how well they separate people, mean waveforms at the extremes of each axis) and fed the filtered signals into the covariance fingerprint.
- Against 6 fixed beta bands (15 to 30 Hz) it wins on 1 to 4 channels: on one channel, log-loss 0.37 vs 0.50.
- The covariance of a filtered signal depends only on the filter's power spectrum, not its shape. Time-reversed and sign-flipped filters, which have identical spectra, give the same scores to within 0.002. So the gain is spectral.
- Against 18 bands of 2 Hz from 4 to 40 Hz, burst filters lose in all 50 channel subsets tested (one channel: 0.37 vs 0.16), and adding them to the 18 bands changes nothing.

## 5. Burst amplitude is the missing piece

Burst amplitude is the most test-retest stable property of sensorimotor beta events (Pauls et al. 2024, Clin Neurophysiol 163:244-254), and the shape features above z-score every waveform, which removes it.

| Cross-day accuracy | C3 | C3 + C4 | 9 channels |
|---|---|---|---|
| Burst shape, z-scored | 0.35 | 0.44 | 0.47 |
| Burst amplitude alone | 0.45 | 0.50 | 0.44 |
| Shape + amplitude | 0.56 | 0.61 | 0.71 |
| Z-scored waveform, same axis selection and tuning as the raw rows | 0.47 | 0.62 | 0.71 |
| Raw waveform, axes by separation | 0.64 | 0.80 | 0.92 |
| Raw waveform, axes by stability | 0.62 | 0.82 | 0.90 |
| 18-band spectrum | 0.94 | 0.98 | 1.00 |

- Keeping the raw waveforms, and choosing PCA axes by how well they separate people, raises cross-day identification with all 9 channels from 0.47 to 0.92 (log-loss 1.05 to 0.25). The z-scored and raw features differ in more than amplitude: the raw ones use 15 PCA axes, a label-based axis choice, and tuned k and C. The z-scored row above gives the z-scored waveforms the same treatment. It reaches 0.71 (log-loss 0.64), about the same as shape plus amplitude. So about half of the jump comes from the axis choice and tuning, and the remaining 0.2 comes from keeping the raw waveform. Identity sits in the combination of shape and amplitude.
- Amplitude alone is not enough. With 9 channels it scores 0.44, below z-scored shape (0.47). It helps most on one or two channels.
- No family loses accuracy between within-session and cross-day tests.
- Added to the 18-band fingerprint by concatenation, burst features give at most small gains (one channel: log-loss 0.180 to 0.163, one test session). I also tried stacking the two models, which scored worse than the 18-band model alone. The stacker was fitted on session-1 probabilities from models trained on one session and applied to probabilities from models trained on two, so I do not read that result as evidence about the features.
- Regularization (C) and the number of axes (k) were tuned by fitting on session 0 and scoring on session 1. The axis ranking itself uses the labels of sessions 0 and 1, so the session-1 scores used for tuning are optimistic. Session 2 enters no fit and is scored once, so the cross-day numbers in the table are not affected.
- The same raw-versus-z-scored gap replicated on a separate clinical dataset recorded in single sessions, which is not public.

## 6. On these channels the motor imagery effect is in mu, not beta

Superlet time-frequency maps (all epochs and 9 channels averaged) show power at 8 to 13 Hz falling by 11.7% during imagery relative to the pre-cue period, while 13 to 30 Hz changes by -0.2%. Each person has a different dominant frequency (for example near 12.5 Hz and near 10 Hz), consistent with the strength of spectral fingerprints in findings 4 and 5.
