# 🌙 Day2Night-GAN — Pix2Pix Image-to-Image Translation

This project implements a **Pix2Pix GAN** that converts **daytime images → nighttime images** using a paired dataset.  

A complete ML pipeline is included:
-  PyTorch training pipeline  
-  Clean U-Net Generator  
-  PatchGAN Discriminator  
-  Streamlit UI for inference  
-  Auto-pairing & renaming utilities  
-  Model checkpoint saving  
-  Clean project structure for portfolios  

---

## 🔥 Demo Results (Upload images manually)

```
Day Image → Generated Night → Real Night
```

Upload sample outputs inside:

```
outputs/
 ├── samples/      # Training sample outputs
 └── inference/    # Prediction results
```

---

## 📂 Project Structure

```txt
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
│   └── app.py                # Streamlit UI
│
├── outputs/
│   ├── samples/              # GAN training outputs
│   ├── inference/            # Generated night images
│   └── G_epoch_10.pth        # Trained model weights
│
├── train_A/                  # Day images (training)
├── train_B/                  # Night images (training)
├── test_A/                   # Day images (testing)
├── test_B/                   # Night images (testing)
│
├── predict.py                # Inference script
├── rename_pairs.py           # Dataset pairing utility
├── .gitignore
└── README.md
```

---

## ⚙️ Model Architecture

### 🌀 Generator — U-Net  
- 8-level encoder–decoder  
- Skip connections  
- Tanh output  

### 🟥 Discriminator — 70×70 PatchGAN  
- Evaluates local patches instead of whole image  
- More stable than full-image discriminator  

---

## 🗂 Dataset

Model expects **paired day–night images**.

Preprocessing used:

- Resize → 150×150  
- RandomCrop → 128×128  
- Normalize → [-1, 1]

Pairing utility:

```
rename_pairs.py
```

---

## 🏋️ Training

Run:

```bash
python training/train.py
```

Outputs:

- Loss values printed in console  
- Samples every 200 steps → `outputs/samples/`
- Checkpoints → `outputs/G_epoch_X.pth`

---

## 🔮 Inference (Prediction)

Generate night version of any day image:

```bash
python predict.py
```

Outputs saved in:

```
outputs/inference/
```

---

## 🌐 Streamlit Web App

Run:

```bash
streamlit run ui/app.py
```

Features:

- Upload any day image  
- View generated night output  
- Download result  
- Clean minimal UI  

---

## 🚀 How to Run

### 1️⃣ Create virtual environment

```bash
python -m venv env
env\Scripts\activate
```
### 2️⃣ Clone the Repository

Clone the project to your local machine:

```bash
git clone https://github.com/Sidhardha36/Day2Night-GAN.git
cd Day2Night-GAN
```

### 3️⃣ Install dependencies

```bash
pip install torch torchvision pillow tqdm streamlit
```

### 4️⃣ Train model

```bash
python training/train.py
```

### 5️⃣ Prediction

```bash
python predict.py
```

### 6️⃣Run the UI

```bash
streamlit run ui/app.py
```

---

## ⭐ Features

-  Stable 128×128 Pix2Pix GAN  
-  Clean U-Net implementation  
-  PatchGAN discriminator  
-  Full training pipeline  
-  Streamlit UI  
-  Ready for resume + GitHub  
-  Real paired dataset  

---

## 🧭 Future Improvements

- 256×256 training  
- Night → Day reverse model  
- Deploy on Streamlit Cloud / HuggingFace  
- Convert to CycleGAN version  

---

## 👤 Author

**Sidhardha Varma**  
B.Tech | Machine Learning Enthusiast  
Day-to-Night GAN — 2025  

---

## 🏆 License

MIT License

