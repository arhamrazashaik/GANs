# FACE\_AGE\_SYNTHESIS\_USING\_GANs

A Python-based implementation for **face age progression and regression** using Generative Adversarial Networks (GANs). This project demonstrates how to synthesize aged faces while preserving identity using a two-stage GAN pipeline.

---

## 🚀 Features

- **Age-conditioned synthesis**: Generate face images at target ages while keeping the same identity.
- **Two-stage GAN architecture**:
  1. **cStyleGAN**: Produces a set of images across multiple ages from a single identity seed.
  2. **FaceGAN (U-Net-based generator)**: Learns finer age details by comparing real and synthesized data.
- **Semi-supervised learning**: Combines real and synthetic images to provide effective paired supervision without relying on large paired datasets.
- **Identity preservation**: Maintains visual cues like facial shape, glasses, beard, across synthesized ages.
- **Dataset support**: UTKFace for training; FG‌-NET for qualitative evaluation.

---

## 📁 Repository Structure

```
FACE_AGE_SYNTHESIS_USING_GANs/
├── python_package/
│   └── notebook/
│       └── example.ipynb        ← Demo notebook showcasing training and inference
├── models/                      ← Pre-trained GAN weights (if available)
├── data/
│   ├── UTKFace/                 ← Training dataset examples
│   └── FG-NET/                  ← Evaluation dataset samples
├── src/
│   ├── cstylegan.py             ← Conditional StyleGAN implementation
│   ├── facegan.py               ← U-Net + adversarial training model
│   ├── utils.py                 ← Data loaders & utility functions
│   └── train.py                 ← Full training script
├── requirements.txt            ← Required Python packages
└── README.md                   ← This file
```

---

## ✅ Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

- Download and place **UTKFace** and **FG-NET** datasets into `data/UTKFace/` and `data/FG-NET/`.

### 3. Train the models

```bash
python src/train.py --dataset UTKFace --mode cStyleGAN
python src/train.py --dataset UTKFace --mode FaceGAN
```

### 4. Run the demo

Open \`\` to walk through inference and visual results.

---

## 🎯 Approach Overview

1. **cStyleGAN**

   - Trains a conditional generator (StyleGAN-like) using age labels and real images.
   - Synthesizes face images across target age groups for the same identity.
   - Provides “paired” age data for further training.

2. **FaceGAN (U-Net)**

   - Uses the synthesized images from cStyleGAN as “ground-truth” for reconstruction.
   - Learns age transformation while preserving high-frequency and identity details.
   - Combines adversarial, identity, and reconstruction losses for training.

3. **Semi-supervised learning**

   - Mixes *real* and *synthesized* training samples.
   - Enhances both age realism and identity permanence without paired data.

---

## 🔍 Example Usage (Notebook)

Within `example.ipynb`, you'll find:

- Loading a real face image + target age
- How cStyleGAN synthesizes multiple age variants
- FaceGAN refinement of aging effects
- Visual comparisons of learned identity preservation and aging progression

---

## ✅ Results & Validation

- Shows realistic aged/rejuvenated faces.
- Maintains identity across long age ranges (e.g. beard, bone structure).
- Quantitative benefits measured via face detection (MTCNN) and recognition models (ResNet‌-50, VGGFace2).

---

## 📒 References

- Wang et al. “Age-Oriented Face Synthesis with Conditional Discriminator Pool…”
  - [arxiv.org](https://arxiv.org/abs/2007.00792)
  - [arxiv.org](https://arxiv.org/abs/1901.07528)
  - [mdpi.com](https://www.mdpi.com/2079-9292/9/4/603)
- **SS-FaceGAN**: Semi-supervised GAN framework combining StyleGAN + U-Net architectures.
- Related works: Pyramid GANs, PFA-GAN, etc.

---

## 🛠 Future Work

- Add more age groups or guide aging paths more precisely.
- Integrate face recognition-loss to further enforce identity through training.
- Optimize inference speed for real-time applications.

---

## 👤 Author

**Arham Raza Shaik**\
[GitHub](https://github.com/arhamrazashaik)

---

## 📄 License

[MIT License](LICENSE) — Free to use, modify, and distribute for academic or commercial purposes.

