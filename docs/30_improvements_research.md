# 30 Research-Backed Improvements for Bat Listener

## Batch A: University / Academic Research (1–10)

1. **Transformer Networks for Multi-Species Classification** — Ulm University: ConvNet-Transformer hybrid classifies entire call sequences, handles overlapping species, 88.9% accuracy on ChiroVox. → `classifier_backend.py`
2. **CNN + Geographic Range Maps (NABat ML)** — Colorado State/USGS: CNN trained on 600K spectrograms, uses range maps as geographic prior to eliminate impossible predictions, 92% accuracy on 30 NA species. → `geo_filter.py`
3. **MobileNetV1 + Grad-CAM Explainability** — University of Tokyo: MobileNetV1 with Bayesian optimization, 98.1% on 30 Japanese species, Grad-CAM visualizes which spectrogram regions drive each prediction. → `classifier_backend.py` + `advanced_spectrogram.py`
4. **Data Augmentation for Acoustic CNNs** — University of East Anglia: Tests 20 spectrogram augmentation methods; frequency/time masking boost accuracy up to 64% with low sample counts. → `classifier_backend.py`
5. **Wavelet Transform for Biosonar Analysis** — Purdue University: CWT with Morlet/matched wavelets provides multiresolution matching bat cochlear processing, sharper ridges for fast FM sweeps. → `advanced_spectrogram.py`
6. **Wigner-Ville Distribution for FM Estimation** — Wrocław University of Technology: Minimizes FM slope standard deviation; reveals Myotis=linear FM, Nyctalus=exponential FM. → `advanced_spectrogram.py`
7. **Cepstral Processing (Cepstrogram)** — UMass Dartmouth/Brown: Windowed short-time cepstral analysis resolves sub-microsecond echo glints below auditory temporal resolution. → `advanced_spectrogram.py`
8. **TFC Isolation with Dynamic Time Warping** — University of New Hampshire: Butterworth HP filter + power-thresholded STFT + DTW + hierarchical clustering isolates individual harmonics within a call, quantifies FM curvature. → `signal_tracking.py`
9. **Tiny CNN for Real-Time Edge Deployment** — American University of Sharjah: ~200K-parameter CNN (~1.5MB), 97.5% accuracy on 8 species, runs on Jetson Nano via TFLite. → `classifier_backend.py`
10. **Particle Filter Frequency Tracking** — University of Edinburgh: Sequential Monte Carlo tracks instantaneous frequency of multiple time-varying components from raw data, auto-detects component count. → `signal_tracking.py`

## Batch B: Open-Source / Practical Signal Processing (11–20)

11. **BatDetect2 Integration** — macaodha/batdetect2: Deep learning model detects/classifies bat calls in full-spectrum audio, pre-trained UK species model, Python API. → `classifier_backend.py`
12. **PCEN (Per-Channel Energy Normalization)** — Lostanlen et al.: Adaptive gain control replaces static log compression, reduces false alarms 5–50× vs log-mel. Params: T=60ms, α=0.8, δ=10, r=0.25. → `advanced_spectrogram.py`
13. **Constant-Q Transform (CQT) Spectrogram** — librosa.cqt: Geometrically-spaced bins (fixed Q=f/Δf), better resolution at low freq, ideal for CF calls. fmin=15kHz, 84 bins. → `advanced_spectrogram.py`
14. **Real-Time Spectral Subtraction** — SoheilGtex/Active-Noise-Cancelling: 20ms frames, 50% overlap-add, EMA noise tracking, per-bin magnitude subtraction with flooring. → `noise_suppression.py`
15. **Wiener Filter Enhancement** — scipy.signal.wiener: Optimal linear MSE gain H(f)=S_signal/(S_signal+S_noise), single-pass, no musical noise artifacts. → `noise_suppression.py`
16. **Bat Call Parameter Extraction** — SonoBat/Montana bat call keys: Extract Fpeak, Fc, Fhi, Flo, duration, upper/lower/total slope, bandwidth, IPI from spectrogram contours. → `call_parameters.py`
17. **GUANO Metadata Embedding** — guano-py: Write GPS, timestamp, species ID, recorder model into WAV headers using GUANO open standard. → `guano_metadata.py`
18. **AudioMoth-Style Amplitude Threshold Triggering** — OpenAcousticDevices: HP filter ~10kHz, per-ms energy, running average over ~500ms, trigger when energy exceeds average by configurable factor for Y consecutive ms. → `triggered_recording.py`
19. **BattyBirdNET-Analyzer Integration** — rdz-oss: BirdNET fine-tuned for bats, covers EU/UK/NA species, 256–384kHz, real-time on RPi. → `classifier_backend.py`
20. **Mel-Scale Spectrogram** — librosa.feature.melspectrogram: 128 mel bands 15–120kHz, 12ms Hann window, 1.5ms hop, compresses high-freq bins, perceptually weighted. → `advanced_spectrogram.py`

## Batch C: Field Biology / Citizen Science (21–30)

21. **GUANO Metadata Standard Compliance** — Wildlife Acoustics/NABat/BatSync: Self-describing recordings with GPS, timestamp, species, recorder info. → `guano_metadata.py` (merged with #17)
22. **EUROBATS/BCT Survey Protocol Modes** — EUROBATS Pub. No.5 / BCT GPG 4th ed: Walked transect, point-count, vehicle transect, static deployment modes with pre-filled metadata and timing rules. → `survey_protocols.py`
23. **Automated Feeding Buzz Detection** — Buzzfindr/BatSpot: Detect terminal-phase echolocation buzzes (rapid pulse sequences) as foraging activity index, F1=0.95. → `activity_metrics.py`
24. **Duty-Cycle-Aware Activity Metrics** — biorxiv 2025.04.15: Call Rate (CR), Activity Index (AI), Bout-Time Percentage (BTP) with confidence flags for duty-cycled recordings. → `survey_protocols.py`
25. **Environmental Covariate Integration** — Gorman et al. 2021: Log temperature, humidity, pressure, moon phase, time-since-sunset as activity correlates; build GLM-style activity models. → `environmental_data.py`
26. **GPS-Tagged Recordings + Range-Limited Species Lists** — NABat/GBIF/IUCN: Filter species list to geographic range overlap, prevents impossible classifications. → `geo_filter.py`
27. **Temporal Pass Plots (TPP)** — Ecol Indicators 2020: Fine-scale bat passes per time interval with sunset/sunrise overlay, comparable between sites/dates. → `activity_metrics.py`
28. **Activity Heatmaps & Seasonal Detection Charts** — BatAMP: Calendar heatmaps (night × species activity), seasonal trend charts, year-over-year phenology. → `activity_metrics.py`
29. **Smartphone Built-In Microphone Support** — Springer 2024: Mobile devices record low-freq bat calls (noctule, serotine) with 0.69–0.90 quality vs professional; auto-detect capability, warn but don't block. → `citizen_science.py`
30. **One-Click Export to Citizen Science Platforms** — Bat2iNat/Somerset Bat Group: Map GUANO→Darwin Core, push to iNaturalist/BatSync/NABat with auto spectrogram thumbnails. → `citizen_science.py`

## New Module Map (12 files)

| New Module | Improvements | Key Dependencies |
|---|---|---|
| `call_parameters.py` | #16 | numpy, scipy |
| `noise_suppression.py` | #14, #15 | numpy, scipy |
| `guano_metadata.py` | #17, #21 | struct (stdlib) |
| `advanced_spectrogram.py` | #12, #13, #20, #5, #6, #7, #3(Grad-CAM) | numpy, scipy |
| `survey_protocols.py` | #22, #24 | stdlib |
| `environmental_data.py` | #25 | stdlib, math |
| `geo_filter.py` | #26, #2(geo prior) | stdlib |
| `activity_metrics.py` | #23, #27, #28 | numpy |
| `triggered_recording.py` | #18 | numpy |
| `signal_tracking.py` | #8, #10 | numpy, scipy |
| `classifier_backend.py` | #1, #4, #9, #11, #19, #21, #24 | numpy (interfaces; models optional) |
| `citizen_science.py` | #29, #30 | stdlib |

All modules implementable in pure Python with numpy/scipy. ML model weights are optional downloads.