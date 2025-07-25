# 🧠 AskToSee

> Interactive vision-language segmentation from **text** and **voice** prompts using CLIP, SAM, and Whisper.

**AskToSee** is an advanced computer vision system that allows users to:
- 🖼 Upload an image
- 💬 Type or speak a natural language prompt (e.g., *"the red car"*)
- 🧠 Automatically segment the object(s) described

This project combines the power of:
- **CLIP** for vision-language understanding
- **SAM (Segment Anything Model)** for high-precision mask generation
- **Whisper** for speech-to-text input
- A clean **Streamlit interface** for easy interaction

---

## 🗂️ Project Structure

## ⚙️ Setup (🪟 Windows)
### 1. Clone the repo:
```bash
   git clone https://github.com/awhtiambou/asktosee.git
   cd asktosee
```
### 2. Run the setup script:
```bash
   setup.bat
```
This will:
 - Create a virtual environment
 - Install PyTorch with CUDA
 - Clone and install CLIP, SAM, and Whisper
 - Install all required dependencies

