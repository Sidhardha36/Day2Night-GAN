📘 README.md — Day2Night-GAN (Pix2Pix Image-to-Image Translation)

🌙 Day → Night Image Translation using Pix2Pix (GAN)

This project implements a Pix2Pix Generative Adversarial Network to convert daytime images into nighttime images using a paired dataset.
It includes:

✔ Full training pipeline (PyTorch)

✔ Clean U-Net Generator + PatchGAN Discriminator

✔ Streamlit Web App for inference

✔ Preprocessing, dataset pairing, renaming utilities

✔ Model checkpoint saving

✔ Visualization outputs

🧠 Overview

Pix2Pix is a conditional GAN that learns a mapping from a source domain (Day) to a target domain (Night).
The model is trained using:

Generator: U-Net (skip connections preserve details)

Discriminator: 70×70 PatchGAN

Loss: Adversarial + L1

This project demonstrates real-world GAN usage and is excellent for portfolios and ML/AI interviews.

🔥 Demo Results (Sample Outputs)

Add your output images here manually on GitHub:

outputs/
    ├── samples/
    └── inference/


Example layout in README:

Day → Generated Night → Real Night


(Upload images and embed them with Markdown)

📂 Project Structure
Day2Night-GAN/
│── models/
│   ├── generator.py
│   └── discriminator.py
│
│── training/
│   ├── train.py
│   ├── dataset.py
│   └── __init__.py
│
│── ui/
│   └── app.py            # Streamlit UI
│
│── outputs/
│   ├── samples/          # Training outputs
│   ├── inference/        # Prediction results
│   ├── G_epoch_10.pth    # Model weights (ignored by Git)
│
│── train_A/              # Day images (training)
│── train_B/              # Night images (training)
│── test_A/               # Day images (testing)
│── test_B/               # Night images (testing)
│
│── predict.py
│── rename_pairs.py
│── .gitignore
│── README.md

⚙️ Model Architecture
Generator: U-Net

8-level encoder–decoder

Skip connections

Output activation: Tanh

Discriminator: 70×70 PatchGAN

Evaluates real vs fake patches

More stable than full-image discriminator

🗂️ Dataset

This project uses Day2Night paired dataset:

Each day image has a matching night version

Images are resized & cropped to 128×128

Custom renaming script to align pairs:

rename_pairs.py

🏋️‍♂️ Training

Run training:

python training/train.py


Outputs:

Loss curves (printed in console)

Generated samples every 200 iterations

Checkpoints saved as:

outputs/G_epoch_X.pth
outputs/D_epoch_X.pth

🔮 Inference (Prediction)

Convert a single daytime image:

python predict.py


Saves output to:

outputs/inference/

🌐 Streamlit Web App

Launch UI:

streamlit run ui/app.py


Features:

Upload a daytime image

See generated nighttime image

Download output

Clean & simple UI

🚀 How to Run the Project
1️⃣ Create venv
python -m venv env
env\Scripts\activate

2️⃣ Install dependencies
pip install torch torchvision pillow tqdm streamlit

3️⃣ Train
python training/train.py

4️⃣ Run inference
python predict.py

5️⃣ Run UI
streamlit run ui/app.py

⭐ Features

✔ Stable 128×128 Pix2Pix GAN

✔ Clean U-Net generator

✔ PatchGAN discriminator

✔ Full training pipeline

✔ Streamlit inference UI

✔ Real paired dataset

✔ Ready for deployment

✔ Perfect for resume/portfolio

🧭 Future Improvements

Train on 256×256 resolution

Add Night → Day model

Deploy on HuggingFace Spaces / Streamlit Cloud

Add cycle consistency (CycleGAN version)

🧑‍💻 Author

Sidhardha Varma
B.Tech | Machine Learning Enthusiast
Day-to-Night GAN Project — 2025

🏆 License

This project is open source under the MIT License.