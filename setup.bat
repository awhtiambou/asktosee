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

REM [5/6] Install third-party libraries from source
echo [5/6] Installing third-party packages from local source...
pip install -e thirdparty\CLIP
pip install -e thirdparty\segment-anything
pip install -e thirdparty\whisper

REM [6/6] Done
echo [6/6] Setup complete!
pause
endlocal