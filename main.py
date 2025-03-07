from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os

app = FastAPI()

# Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create an OpenCV super-resolution instance
sr = cv2.dnn_superres.DnnSuperResImpl_create()

# Load the EDSR model from the weights folder
model_path = "weights/EDSR_x4.pb"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

sr.readModel(model_path)
sr.setModel("edsr", 4)  # 4x upscaling

@app.post("/upscale")
async def upscale(image: UploadFile = File(...)):
    try:
        print("Received request to upscale an image")
        contents = await image.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Upscale the image using EDSR
        output = sr.upsample(img)
        print(f"Input image dimensions: {img.shape}")
        print(f"Output image dimensions: {output.shape}")

        _, buffer = cv2.imencode(".png", output)
        return Response(content=buffer.tobytes(), media_type="image/png")
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
