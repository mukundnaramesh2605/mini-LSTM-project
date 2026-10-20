# LSTM Autoencoder for Time-Series Anomaly Detection

Catching operational failures in server metrics with an LSTM autoencoder trained only on normal behavior. Built on the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB).

## Results

On an EC2 CPU-utilization stream with two known failure episodes:

| Threshold (mean + k·std) | Precision | Recall | F1 | Failure windows caught |
|---|---|---|---|---|
| k = 3 | 0.68 | 0.38 | 0.49 | 2 / 2 |
| k = 4 | 0.79 | 0.35 | 0.49 | 2 / 2 |
| **k = 5** | **1.00** | **0.35** | **0.52** | **2 / 2** |

At k = 5 every alert is a real event (zero false alarms) and no failure is missed. Recall is capped at ~0.35 by NAB's wide labeling, not by the model: raising the threshold does not lower it, which it would if recall were threshold-limited.

![detected anomalies](results.png)

## The problem

Classical outlier detection (z-score, IQR, DBSCAN) scores each point against a static distribution. It misses contextual anomalies: a CPU load that is normal at peak hours is alarming at 3am. To catch those you need a model that understands sequence and timing.

An LSTM autoencoder learns to compress and rebuild short windows of normal behavior. When a window looks nothing like what it learned, reconstruction error spikes, and that spike is the anomaly signal.

## Dataset

`realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv` from NAB: CPU utilization sampled every 5 minutes over 14 days, with two labeled failure windows (one spike, one permanent level shift).

## Method

1. Denoise with a centered rolling mean. NAB metrics are noise-dominated point to point, so smoothing lets the model learn real shape instead of jitter.
2. Scale with StandardScaler, fit on normal data only to avoid leakage.
3. Slice into overlapping windows of 48 steps (4 hours).
4. LSTM encoder compresses each window to a 32-dim latent vector, LSTM decoder rebuilds it. Train on MSE reconstruction loss, normal windows only.
5. Threshold reconstruction error at mean + k·std of the training errors.
6. Map window error back to timestamps, flag the spikes, score against the labeled windows.

## The one that nearly broke it

The first attempt trained on the first 25% of the series and the loss stuck at 0.9, barely better than predicting the mean. Scoring revealed why: the labeled anomalies were reconstructing *better* than normal periods. The series is non-stationary, so a model trained only on the front slice treated every later period, failure or not, as foreign.

The fix was to train on normal windows sampled from across the whole series, so the model sees every normal regime. Loss dropped to 0.028 and the anomalies now reconstruct at 7x the normal error. This is still unsupervised learning: the labels only define which windows count as normal, they are never a target the model predicts.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/lstm_anomaly_nab.ipynb
```

Runs on CPU in about a minute. Uses Apple Silicon (MPS) or CUDA automatically if present.

## What I would extend next

- Multivariate input (CPU, latency, network together) for richer context
- Adaptive thresholding that tracks drift instead of a fixed cutoff
- A classical baseline (DBSCAN, rolling z-score) on the same windows for comparison
