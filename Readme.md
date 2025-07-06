# 👤 Face Age Synthesis using GANs (MTLFace)

This project uses a **Conditional GAN**–based **MTLFace** (Multi-Task Learning Face) model to perform:

- 🔍 Face Verification  
- 🧠 Age Estimation  
- 🕒 Face Age Progression & Regression (Aging GAN)  

It takes a face image and generates realistic age-transformed versions (from young to old) while **preserving identity**.

---

## 📌 Features

- ✅ Multi-task Learning Model (FaceID + Age Estimation + Age Synthesis)
- ✅ Conditional GAN trained to generate face at specific target ages
- ✅ Architecture inspired by StyleGAN2 for high-quality synthesis
- ✅ Cosine Similarity for face recognition
- ✅ PyTorch-based and GPU-accelerated

---

## 🧰 Tech Stack

- Python 🐍  
- PyTorch ⚡  
- MTLFace Pretrained Model  
- TorchVision (for transforms and visualization)  
- Google Colab or local CUDA environment  

---

## 📁 Project Structure

├── MTLFace/ # Cloned repo
├── mtlface_checkpoints.tar # Pretrained weights
├── 230302.png, 071A42.JPG # Sample face images
├── inference_script.py # Your implementation
└── README.md # This file




---

## 🔧 Setup Instructions

1. **Clone the MTLFace repo**
```bash
git clone --depth 1 https://github.com/Hzzone/MTLFace.git
mv MTLFace/python_package/* .
Install dependencies


pip install -U --no-cache-dir gdown --pre
pip install Ninja torch torchvision
Download pretrained weights

gdown --id 1OmfAjP3BAqVxaQ2pwyJuOYUHy_incMNd -O mtlface_checkpoints.tar
Fix imports for StyleGAN2 ops


echo "from mtlface.stylegan2.op import upfirdn2d, FusedLeakyReLU, fused_leaky_relu" > colab_init.py
python colab_init.py
🚀 Running the Project
🖼️ Preprocess Images


from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

img = Image.open('230302.png').convert("RGB")
input_img = transform(img).unsqueeze(0).cuda()
🔍 Load the Model


from mtlface.modules import MTLFace
mtlface = MTLFace().cuda().eval()
mtlface.load_state_dict(torch.load('mtlface_checkpoints.tar'))
✅ Face Verification + Age Estimation

x_vec, x_age = mtlface.encode(input_img)
# x_vec: feature vector for face verification
# x_age: predicted age
🧓 Face Age Synthesis

bs = input_img.size(0)
target_labels = torch.arange(7).cuda().unsqueeze(1).repeat(bs, 1).flatten()
repeat_images = input_img.unsqueeze(1).repeat(1, 7, 1, 1, 1).view(-1, 3, 112, 112)
outputs = mtlface.aging(repeat_images, target_labels).view(bs, 7, 3, 112, 112)
🎨 Visualize Results

from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

grid = make_grid(torch.cat([input_img.unsqueeze(1), outputs], dim=1).view(-1, 3, 112, 112)) * 0.5 + 0.5
to_pil_image(grid).show()
🧠 Key Concepts
Conditional GAN (cGAN): Generator takes a face + age label → outputs age-transformed face

Multi-task Learning: Trains for recognition, estimation, and generation together

Cosine Similarity: Used to compare face identity feature vectors

StyleGAN2-Inspired Layers: Architecture uses advanced layers like FusedLeakyReLU for quality

🎯 Applications
Missing person search (age progression)

Digital avatars & social apps

Face morphing and entertainment

Identity-preserving image editing

📚 References
MTLFace GitHub

StyleGAN2 Paper

Face Aging using GANs – Survey

👨‍💻 Author
Arham Raza
B.Tech CSE | AI & ML Enthusiast
