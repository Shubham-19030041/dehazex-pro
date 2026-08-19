DeHazeX Pro

Image dehazing web application comparing three deep learning architectures — AOD-Net (CNN), Attention U-Net, and DehazeGAN — alongside the classical Dark Channel Prior (DCP) algorithm. Built with TensorFlow/Keras and Flask, trained on the RESIDE benchmark dataset.

Features
Upload any hazy/foggy image and remove haze using a selected model
Compare all 4 methods (3 deep learning + 1 classical) side by side on the same image
Automatic weather/haze-density detection with model auto-selection
Live quality metrics: PSNR, SSIM, Entropy, Edge Preservation, inference time
Model comparison dashboard (evaluated on the RESIDE SOTS test set)
Tech Stack
Deep learning: TensorFlow / Keras
Backend: Flask
Frontend: HTML, CSS, JavaScript
Image processing: OpenCV, scikit-image, Pillow
Dataset: RESIDE (Realistic Single Image DEhazing) benchmark
Models
Model	Type	Notes
AOD-Net	CNN	Lightweight, fastest inference
Attention U-Net	Encoder-decoder with attention gates	Best PSNR/SSIM among the trained models
DehazeGAN	Generative Adversarial Network	Generator + PatchGAN discriminator
DCP	Classical (no training)	Dark Channel Prior, He et al. 2009
Running Locally

1. Install dependencies ```bash python -m venv venv venv\Scripts\activate # Windows

source venv/bin/activate # macOS/Linux

pip install -r requirements.txt ```

2. Prepare the dataset (optional, only needed to retrain models)

Place the RESIDE dataset so it matches this structure: ``` dataset/ITS/hazy/ dataset/ITS/clear/ dataset/SOTS/hazy/ dataset/SOTS/clear/ ``` Or use `setup_dataset.py` to map a Kaggle-style download automatically: ```bash python setup_dataset.py --root "/path/to/your/dataset" ```

3. Train models (optional — pretrained models are already included in `saved_models/`) ```bash python train.py --model aod_net --epochs 20 --batch_size 4 python train.py --model attention_unet --epochs 5 --batch_size 4 python train.py --model gan --epochs 5 --batch_size 4 ```

4. Evaluate models on the test set ```bash python evaluate.py ```

5. Run the web app ```bash python web_app.py ``` Open `http://127.0.0.1:5000\` in your browser.

Project Structure

``` ├── models.py # Model architectures (AOD-Net, Attention U-Net, GAN) ├── dark_channel_prior.py # Classical DCP baseline ├── utils.py # Metrics, dataset loading, weather detection ├── train.py # Training script ├── evaluate.py # Test-set evaluation and comparison table ├── web_app.py # Flask web application ├── templates/index.html # Frontend UI ├── static/ # CSS and JavaScript └── saved_models/ # Trained model weights (.keras) ```

Deployment

This app is deployable to platforms like Render (backend) via the included `Procfile` and `requirements-web.txt`.