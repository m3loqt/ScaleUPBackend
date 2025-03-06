from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import torch
from Real_ESRGAN.realesrgan import RealESRGANer
from Real_ESRGAN.realesrgan.archs.srvgg_arch import SRVGGNetCompact

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Real-ESRGAN
model = RealESRGANer(
    scale=4,  # Upscale factor
    model_path="weights/realesr-general-x4v3.pth",
    model=SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'),
    tile=0,  # Tile size (0 for no tiling)
    tile_pad=10,
    pre_pad=0,
    half=False,  # Use FP16 (faster but less accurate)
)

@app.post("/upscale")
async def upscale(image: UploadFile = File(...)):
    try:
        print("Received request to upscale an image")

        # Read image bytes
        contents = await image.read()
        img_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print("Error: Invalid image file")
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Resize image if it's too large
        max_size = 1024  # Maximum dimension (width or height)
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # Convert image to RGB (Real-ESRGAN expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Upscale image using Real-ESRGAN
        print("Upscaling image...")
        output, _ = model.enhance(img, outscale=4)  # Upscale by 4x
        print(f"Input image dimensions: {img.shape}")
        print(f"Output image dimensions: {output.shape}")
        print("Image successfully upscaled")

        # Convert back to BGR for OpenCV
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Convert image back to bytes
        _, buffer = cv2.imencode(".png", output)
        return Response(content=buffer.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")