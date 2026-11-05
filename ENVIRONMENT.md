# Reference Environment

All reported experiments were run with:
```
python: 3.10.12
torch: 2.9.0+cu128 (built for CUDA 12.8)
numpy: 2.2.6
pandas: 2.3.3
scipy: 1.15.3
matplotlib: 3.10.8
nibabel: 5.3.3
SimpleITK: 2.5.3
```

OS: Linux x86_64. Training on CUDA GPUs (bf16 AMP); evaluation/aggregation CPU-capable.
Install via `pip install -r requirements.txt` (pinned minor versions).
