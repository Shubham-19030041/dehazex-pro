# DeHazeX Pro — TensorFlow + Flask Rebuild (Step-by-Step)

This is your FYP rebuilt with **TensorFlow/Keras** (instead of PyTorch) and a
**Flask web app** matching your screenshots (dark theme, Single Image tab,
All 3 Models compare, live metrics). No PyCharm needed — use VS Code, or even
just a terminal.

```
DeHazeX_Pro/
├── requirements.txt
├── models.py          # AOD-Net, Attention U-Net, GAN (TensorFlow)
├── utils.py            # metrics, dataset loading, weather detection
├── train.py             # trains any of the 3 models
├── evaluate.py          # compares all 3 on the test set → CSV/Excel/chart
├── web_app.py            # Flask server (this is what you demo)
├── templates/index.html
├── static/css/style.css
├── static/js/script.js
├── dataset/ITS/{hazy,clear}     # training pairs go here
├── dataset/SOTS/{hazy,clear}    # test pairs go here
└── saved_models/                # trained .keras files land here
```

## Step 1 — Install Python & set up the environment

1. Install **Python 3.10 or 3.11** (TensorFlow doesn't yet fully support 3.12+ on all platforms).
2. Open a terminal in the `DeHazeX_Pro` folder (in VS Code: `Terminal > New Terminal`).
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Step 2 — Get the RESIDE dataset

Your report uses the RESIDE benchmark (ITS = training, SOTS = testing).

**Already downloaded it from Kaggle?** Kaggle RESIDE mirrors usually extract to:
```
<root>/train/hazy/...
<root>/train/gt/...
<root>/test/hazy/...
<root>/test/gt/...
```
Run the included helper to map that straight into what this project expects
(`dataset/ITS/hazy`, `dataset/ITS/clear`, `dataset/SOTS/hazy`, `dataset/SOTS/clear`):

```bash
python setup_dataset.py --root "/path/to/your/extracted/reside-folder"
```

This creates symlinks by default (instant, no extra disk space). If your
system can't create symlinks (e.g. Windows without developer mode/admin),
add `--copy` to physically copy the files instead:
```bash
python setup_dataset.py --root "/path/to/your/extracted/reside-folder" --copy
```

It prints a file count for each of the 4 folders at the end so you can
confirm everything mapped correctly — e.g.:
```
dataset/ITS/hazy: 13990 files
dataset/ITS/clear: 1399 files
dataset/SOTS/hazy: 500 files
dataset/SOTS/clear: 500 files
```
(It's normal for `clear`/`gt` to have far fewer files than `hazy` — RESIDE
generates many hazy variants per single clear/ground-truth image.)

**Starting from scratch instead?** Search "RESIDE dataset image dehazing" —
it's on the official project page and several Kaggle mirrors (search "RESIDE
ITS SOTS Kaggle"). Download the **ITS (Indoor Training Set)** and **SOTS
(Synthetic Objective Testing Set)**, then use `setup_dataset.py` the same way.

**Don't have time to process the full dataset?** You can point `--root` at
a folder containing just a subset (e.g. 200–500 pairs) to prove the pipeline
works and get real (if less polished) metrics for your report — training
will just be much faster.

## Step 3 — Sanity-check the models

Before training, just confirm the architectures build correctly:
```bash
python models.py
```
You should see parameter counts for each model (AOD-Net ~ tens of thousands,
Attention U-Net ~ millions, GAN generator ~ millions) — this matches the
"Parameters" figures in your viva guide.

## Step 4 — Train each model

Train one at a time (each writes a `.keras` file to `saved_models/` and a
loss-curve PNG to `outputs/`):

```bash
python train.py --model aod_net --epochs 20 --batch_size 4
python train.py --model attention_unet --epochs 20 --batch_size 4
python train.py --model gan --epochs 20 --batch_size 8
```

Notes:
- On CPU, Attention U-Net and GAN will take a while for 20 epochs on 6000
  images — if you're short on time, reduce `--epochs` (e.g. 5–8) to get a
  working demo; just say so honestly if asked in viva (the guide's expected
  results assume the full 20 epochs).
- If you have an NVIDIA GPU, `pip install tensorflow[and-cuda]` instead and
  TensorFlow will use it automatically — training will be dramatically faster.

## Step 5 — Evaluate & generate the comparison table

Once all three are trained:
```bash
python evaluate.py
```
This runs all 3 models on `dataset/SOTS`, computes PSNR/SSIM/Entropy/Edge/
Time for each, and writes:
- `outputs/compare/comparison.csv`
- `outputs/compare/comparison.xlsx`  (for your report/Excel export)
- `outputs/compare/comparison_bar_chart.png`

## Step 6 — Run the web app (matches your screenshots)

```bash
python web_app.py
```
Open **http://127.0.0.1:5000** in your browser. You'll see:
- **Single Image tab**: upload a hazy photo, pick a model (or check
  "Auto Select"), click **Dehaze**, see the live metrics update.
- **All 3 Models** button: runs all three and shows them side by side with
  PSNR tags, just like your screenshot.
- **Save**: downloads the dehazed PNG.

If a model hasn't been trained yet, the app tells you exactly which
`train.py` command to run instead of crashing.

## Step 7 — Extending it (optional, matches the "8 tabs" in your report)

The core (Single Image + Compare) is fully working. To match every tab from
the original guide, add incrementally — each is a self-contained addition:
- **Batch tab**: loop `run_model()` over a folder of images.
- **Webcam tab**: use `navigator.mediaDevices.getUserMedia` in JS + a
  `/api/dehaze` call per frame (or WebSocket for smoother streaming).
- **Video tab**: use OpenCV (`cv2.VideoCapture`) to read frames, dehaze each
  with the loaded Keras model, write out with `cv2.VideoWriter`.
- **YOLO tab**: `pip install ultralytics`, run `YOLO('yolov8n.pt')(dehazed_img)`
  after dehazing, draw boxes, show hazy vs. dehazed vs. detections.

Ask me for code for any of these next — I can build them the same way.

## Step 8 — Viva prep

Your PDF's **Sections 12–16** (60+ Q&A and exam-day checklist) still apply
word-for-word — only the framework changed (TensorFlow instead of PyTorch),
not the architectures, math, or metrics. Just swap any mention of "PyTorch"
for "TensorFlow/Keras" if asked directly.
