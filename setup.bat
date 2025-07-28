@echo off
setlocal
echo === AskToSee Setup Script ===

REM [1/6] Create virtual environment
echo [1/6] Creating virtual environment...
python -m venv .venv

REM [2/6] Activate venv
echo [2/6] Activating virtual environment...
call .venv\Scripts\activate

REM [3/6] Upgrade pip and install base packages
echo [3/6] Installing core packages...
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install opencv-python matplotlib Pillow streamlit transformers
pip install git+https://github.com/facebookresearch/segment-anything.git

REM [4/6] Clone third-party repos if not already cloned
echo [4/6] Cloning third-party libraries...
IF NOT EXIST "thirdparty" mkdir thirdparty
cd thirdparty

IF NOT EXIST "CLIP" (
    git clone https://github.com/openai/CLIP.git
) ELSE (
    echo - CLIP already cloned
)

IF NOT EXIST "segment-anything" (
    git clone https://github.com/facebookresearch/segment-anything.git
) ELSE (
    echo - segment-anything already cloned
)

IF NOT EXIST "whisper" (
    git clone https://github.com/openai/whisper.git
) ELSE (
    echo - whisper already cloned
)

cd ..

REM [4b/6] Download SAM checkpoint if missing
echo [4b/6] Checking for SAM model checkpoint...
IF NOT EXIST "thirdparty\segment-anything\weights\sam_vit_b.pth" (
    echo - Downloading sam_vit_b.pth...
    powershell -Command "New-Item -ItemType Directory -Path 'thirdparty/segment-anything/weights'"
    powershell -Command "Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile 'thirdparty\segment-anything\weights\sam_vit_b.pth'"
) ELSE (
    echo - sam_vit_b.pth already exists
)

REM [5/6] Install third-party libraries from source
echo [5/6] Installing third-party packages from local source...
pip install -e thirdparty\CLIP
pip install -e thirdparty\whisper

REM [6/6] Done
echo [6/6] Setup complete!
pause
endlocal