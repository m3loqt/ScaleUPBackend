from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific origins in production (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "FastAPI Backend is Running!"}

@app.post("/upscale")
async def upscale(image: UploadFile = File(...)):
    try:
        print("Received request to upscale an image")

        # Read image bytes
        contents = await image.read()
        print(f"Image size: {len(contents)} bytes")

        # Convert bytes to numpy array
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print("Error: Invalid image file")
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Upscale image (2x)
        upscaled_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Convert image back to bytes
        _, buffer = cv2.imencode(".png", upscaled_img)
        print("Image successfully upscaled")

        return Response(content=buffer.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
