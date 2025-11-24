📘 Day2Night-GAN — Pix2Pix Image-to-Image Translation
🌙 Convert Day Images → Night Images using GANs (Pix2Pix)

This project implements a Pix2Pix Generative Adversarial Network that converts daytime images into nighttime images using paired training data.

It includes:

- Full PyTorch training pipeline
- Clean U-Net Generator
- PatchGAN Discriminator
- Streamlit Web App for inference
- Dataset pairing + renaming utilities
- Model checkpoint saving
- Clear visualization outputs

🧠 Overview

Pix2Pix is a conditional GAN that learns a mapping:

Day Image  →  Generator  →  Night Image


Key components:

Generator: U-Net (skip connections preserve spatial detail)

Discriminator: 70×70 PatchGAN

Loss: Adversarial Loss + L1 Loss

Input size: 128×128



🔥 Demo Results (Sample Outputs)

Upload your generated results manually into your repo:

outputs/
 ├── samples/      # GAN training snapshots
 └── inference/    # Final prediction results


Example layout in README:

Day Image → Generated Night → Real Night


(You can embed images manually using Markdown)

📂 Project Structure
Day2Night-GAN/
│
├── models/
│   ├── generator.py
│   └── discriminator.py
│
├── training/
│   ├── train.py
│   ├── dataset.py
│   └── __init__.py
│
├── ui/
│   └── app.py                  # Streamlit UI
│
├── outputs/
│   ├── samples/                # Training samples
│   ├── inference/              # Prediction results
│   └── G_epoch_10.pth          # Model weights (ignored by Git)
│
├── train_A/                    # Day images (training)
├── train_B/                    # Night images (training)
├── test_A/                     # Day images (testing)
├── test_B/                     # Night images (testing)
│
├── predict.py                  # Inference script
├── rename_pairs.py             # Dataset file pairing utility
│
├── .gitignore
└── README.md

⚙️ Model Architecture
🏗 Generator — U-Net

8-level encoder–decoder

Skip connections

Output activation: Tanh

Works well for image→image translation

🔍 Discriminator — 70×70 PatchGAN

Classifies local patches instead of entire image

More stable than full-image discriminator

Produces the “Patch” realism map

🗂 Dataset

Uses a paired Day/Night dataset:

Each Day image has a matching Night image

Preprocessing steps:
1.Resize (150,150)
2.RandomCrop (128)
3.Normalize

Custom renaming script ensures filenames match:

rename_pairs.py

🏋️‍♂️ Training the Model

Run training:

python training/train.py


Training outputs:

Loss prints (Generator & Discriminator)

Sample results saved every 200 batches

Model checkpoints:

outputs/G_epoch_X.pth
outputs/D_epoch_X.pth

🔮 Inference (Prediction)

Generate night image from a day image:

python predict.py


Result saved in:

outputs/inference/

🌐 Streamlit Web App

Launch UI:

streamlit run ui/app.py


Features:

Upload daytime image

Generate nighttime image

Download final result

Clean modern interface

🚀 How to Run the Project
1️⃣ Create virtual environment
python -m venv env
env\Scripts\activate

2️⃣ Install dependencies
pip install torch torchvision pillow tqdm streamlit

3️⃣ Train the model
python training/train.py

4️⃣ Run inference
python predict.py

5️⃣ Run Streamlit UI
streamlit run ui/app.py

⭐ Features

1. Stable 128×128 Pix2Pix GAN
2. Clean U-Net generator
3. PatchGAN discriminator
4. Full training pipeline
5. Streamlit web interface
6. Works on real paired dataset
7. Perfect for portfolio & resume

🧭 Future Improvements

🔹 Upgrade training to 256×256 resolution
🔹 Add reverse model: Night → Day
🔹 Deploy on HuggingFace Spaces / Streamlit Cloud
🔹 Add CycleGAN version
🔹 Add gradient penalty / training stabilization

👨‍💻 Author

Sidhardha Varma
B.Tech — Machine Learning Enthusiast
Day-to-Night GAN Project (2025)

🏆 License

This project is open-source under the MIT License.