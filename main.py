from fastapi import FastAPI, File, UploadFile, Response, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import torch
from realesrgan import RealESRGAN

app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your trusted domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Real‑ESRGAN model for 4x upscaling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_esr = RealESRGAN(device, scale=4)
model_esr.load_weights("RealESRGAN_x4.pth")  # Ensure the weights file is present

@app.get("/")
def home():
    return {"message": "FastAPI Backend is Running!"}

@app.post("/upscale")
async def upscale(image: UploadFile = File(...), scale: int = Form(4)):
    try:
        print("Received request to upscale an image with scale factor:", scale)
        # Read image bytes
        contents = await image.read()
        print(f"Image size: {len(contents)} bytes")
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print("Error: Invalid image file")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process based on selected scale factor
        if scale == 4:
            # Convert image from BGR to RGB for Real‑ESRGAN
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # AI-enhanced upscaling using Real‑ESRGAN
            upscaled_rgb = model_esr.predict(img_rgb)
            # Convert back to BGR for encoding
            upscaled_img = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)
        elif scale == 2:
            # Basic 2x upscaling using interpolation
            upscaled_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        else:
            # If an unsupported scale is passed, return the original image
            upscaled_img = img

        # Encode image to PNG format
        _, buffer = cv2.imencode(".png", upscaled_img)
        print("Image successfully upscaled")
        return Response(content=buffer.tobytes(), media_type="image/png")
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
