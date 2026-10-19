# Facial Expression Recognition (Minimal)
---
title: AI Facial Expression Recognition
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---



Quick start

1. Create a Python virtual env and activate it.
2. Install dependencies: `pip install -r requirements.txt`.
3. Place `fer2013.csv` in a `data/` folder (download from public sources).
4. Train: `python train.py --data data/fer2013.csv --epochs 30`.
5. Run inference: `python inference.py --model models/fer_model.h5`.
