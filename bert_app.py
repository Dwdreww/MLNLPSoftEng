import subprocess
import sys

def install(package, index_url=None):
    """Install a package using pip, optionally specifying an index URL."""
    cmd = [sys.executable, "-m", "pip", "install"] + package.split()
    if index_url:
        cmd.extend(["--index-url", index_url])
    subprocess.check_call(cmd)

print("\n[1/4] Detecting NVIDIA GPU...")

has_cuda = False
try:
    result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        has_cuda = True
except Exception:
    has_cuda = False

# -------------------------------
# Install PyTorch (GPU or CPU)
# -------------------------------
if has_cuda:
    print("[INFO] NVIDIA GPU detected! Installing PyTorch 2.7.1 + CUDA 11.8...")
    install("torch==2.7.1+cu118", "https://download.pytorch.org/whl/cu118")
    install("torchvision==0.22.1+cu118", "https://download.pytorch.org/whl/cu118")
    install("torchaudio==2.7.1", "https://download.pytorch.org/whl/cu118")
else:
    print("[INFO] No GPU detected. Installing CPU-only PyTorch...")
    install("torch")
    install("torchvision")
    install("torchaudio")

# -------------------------------
# Install OCR dependencies
# -------------------------------
print("\n[2/4] Installing EasyOCR + OpenCV + Pillow...")
install("easyocr")
install("opencv-python")
install("pillow")

# -------------------------------
# Install Gradio
# -------------------------------
print("\n[3/4] Installing Gradio...")
install("gradio")

print("\n[4/4] DONE! All required dependencies installed successfully.\n")
