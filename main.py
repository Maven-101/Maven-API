from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import numpy as np
from deepface import DeepFace
import os
import uuid


app = FastAPI()

# Optional: Allow frontend (e.g., React/Vite) to access this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract-embedding/")
async def extract_embedding(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        temp_path = f"temp_{file_id}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get embedding
        embedding = DeepFace.represent(img_path=temp_path, model_name="Facenet", detector_backend='mtcnn')[0]['embedding']
        os.remove(temp_path)

        # Convert to float32 and return as list
        embedding = np.array(embedding).astype(np.float32).tolist()
        return JSONResponse(content={"embedding": embedding})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
