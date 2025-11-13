# install_dependencies.py

import subprocess
import sys

def install(package, index_url=None):
    """Install a package using pip, optionally specifying an index URL."""
    cmd = [sys.executable, "-m", "pip", "install", package]
    if index_url:
        cmd.extend(["--index-url", index_url])
    subprocess.check_call(cmd)

# -----------------------------
# 0️⃣ Upgrade pip, setuptools, wheel first
# -----------------------------
print("\n[INFO] Upgrading pip, setuptools, and wheel...")
install("pip --upgrade")
install("setuptools --upgrade")
install("wheel --upgrade")

# -----------------------------
# 1️⃣ Core Python libraries
# -----------------------------
print("\n[INFO] Installing core Python libraries...")
install("numpy")
install("pandas")
install("matplotlib")
install("seaborn")
install("tqdm")
install("scikit-learn")

# -----------------------------
# 2️⃣ Tokenizers & Transformers (prebuilt wheels, avoids Rust)
# -----------------------------
print("\n[INFO] Installing Transformers and Tokenizers...")
install("tokenizers==0.13.3")         # prebuilt wheel
install("transformers==4.30.2")       # compatible with tokenizers

# -----------------------------
# 3️⃣ Detect GPU and install PyTorch accordingly
# -----------------------------
print("\n[INFO] Detecting NVIDIA GPU for PyTorch installation...")

has_cuda = False
try:
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        has_cuda = True
except Exception:
    has_cuda = False

if has_cuda:
    print("[INFO] NVIDIA GPU detected! Installing PyTorch 2.7.1 + CUDA 11.8 (RTX 2060 compatible)...")
    install("torch==2.7.1+cu118", "https://download.pytorch.org/whl/cu118")
    install("torchvision==0.22.1+cu118", "https://download.pytorch.org/whl/cu118")
    install("torchaudio==2.7.1", "https://download.pytorch.org/whl/cu118")
else:
    print("[INFO] No NVIDIA GPU detected. Installing CPU-only PyTorch...")
    install("torch")
    install("torchvision")
    install("torchaudio")

# -----------------------------
# 4️⃣ Gradio for deployment
# -----------------------------
print("\n[INFO] Installing Gradio for deployment...")
install("gradio")

print("\n✅ All dependencies installed successfully!")
