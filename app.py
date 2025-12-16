from fastapi import FastAPI, UploadFile, File
import uuid
import shutil
import os
from fuel_detector import get_fuel_level

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/fuel-level")
async def fuel_level(file: UploadFile = File(...)):
    # Save uploaded image
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO logic
    result = get_fuel_level(filepath)

    # Optional: delete file after processing
    os.remove(filepath)

    return {
        "success": True,
        "result": result
    }
