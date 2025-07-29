from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, base64, os
import numpy as np

from core.segmenter import PromptSegmenter
from models.whisper_wrapper import WhisperTranscriber

# Initialize models
app = FastAPI(title="AskToSee API")
segmenter = PromptSegmenter()
transcriber = WhisperTranscriber()

# Optional CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Change to client address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "API is up and running!"}

@app.post("/segment")
async def segment_text(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        mask, score = segmenter.run(pil_image, prompt)

        if mask is None:
            return JSONResponse(status_code=404, content={"message": "No object matched the prompt."})

        # Convert mask to base64
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        buffer = io.BytesIO()
        mask_img.save(buffer, format="PNG")
        encoded_mask = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "prompt": prompt,
            "score": round(score, 4),
            "mask": encoded_mask,
            "message": "Segmented successfully"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/segment/audio")
async def segment_audio(
    image: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    try:
        # Save audio temporarily
        ext = audio.filename.split('.')[-1]
        temp_audio_path = f"temp_audio.{ext}"
        with open(temp_audio_path, "wb") as f:
            f.write(await audio.read())

        # Transcribe
        prompt = transcriber.transcribe(temp_audio_path)
        os.remove(temp_audio_path)

        # Run segmentation
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask, score = segmenter.run(pil_image, prompt)

        if mask is None:
            return JSONResponse(status_code=404, content={"message": "No object matched the prompt."})

        # Encode mask
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        buffer = io.BytesIO()
        mask_img.save(buffer, format="PNG")
        encoded_mask = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "prompt": prompt,
            "score": round(score, 4),
            "mask": encoded_mask,
            "message": "Segmented successfully"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
