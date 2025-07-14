# 🎨 Image Style Transfer using CycleGAN

> Built with the tools and technologies:  
> ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
> ![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
> ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

---

This project implements an image style transfer pipeline using CycleGAN — a type of Generative Adversarial Network (GAN) that learns to translate images from one domain to another without needing paired examples. It supports both notebook/script-based training and a user-friendly Streamlit web app for real-time image transformation.

---

## 🧠 Project Highlights

- Unpaired Image-to-Image translation using CycleGAN
- Train and test CycleGAN with Jupyter or Python scripts
- Real-time image style transfer via Streamlit interface 🌐
- Progress tracking with `tqdm`, visualizations with `matplotlib`
- Enhanced image processing using `Pillow`, `NumPy`

---

## 🗂️ Project Structure

Image Style Transfer (CycleGAN)/
├── notebooks/
│ └── CycleGAN_project.ipynb # Training and experiments (Jupyter)
├── scripts/
│ └── CycleGAN_project.py # Training/testing script
├── streamlit_app.py # 🌐 Streamlit app for real-time use
├── requirements.txt # Project dependencies
├── .gitignore
└── .git/

---

## 📦 Tech Stack

- Python 3.10
- TensorFlow 2.10.1 + Keras
- TensorFlow Addons
- NumPy, Matplotlib, Pillow
- tqdm, IPython
- **Streamlit** for interactive UI
- `skillsnetwork` (for monitoring/training utils)

---

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/image-style-transfer-cyclegan.git
   cd image-style-transfer-cyclegan
   ```

2. **(Optional) Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate

    ```

3. **Navigate to the project directory:**

    ```bash
    cd BlogVerse
    ```

4. **Install the dependencies:**

    Using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

---

### 🧪 Usage

🧑‍💻 Option 1: Jupyter Notebook

    jupyter notebook notebooks/CycleGAN_project.ipynb

🐍 Option 2: Python Script

    python scripts/CycleGAN_project.py
    
🌐 Option 3: Streamlit App

    streamlit run streamlit_app.py


