from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

# Add CORS middleware so that requests from any origin are allowed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your Stability API key – ideally, load from an environment variable.
STABILITY_API_KEY = "sk-waClzCnCc5xPpk5PD6S0OdhcpVOA2qcy0njsnhf25w9Yy400"

@app.post("/upscale")
async def upscale(image: UploadFile = File(...)):
    try:
        print("Received request to upscale an image")
        # Read the image bytes from the uploaded file
        contents = await image.read()

        # Prepare the payload for the external upscaling API
        files = {
            "image": ("image.png", contents, image.content_type)
        }
        data = {
            "prompt": "Enhance this image with clarity and detail",  # You can adjust the prompt as needed
            "output_format": "png"
        }
        headers = {
            "authorization": f"Bearer {STABILITY_API_KEY}",
            "accept": "image/*"
        }

        # External API endpoint from Stability
        api_url = "https://api.stability.ai/v2beta/stable-image/upscale/conservative"

        # Make the POST request to the external API
        response = requests.post(api_url, headers=headers, files=files, data=data)
        
        if response.status_code == 200:
            print("Image successfully upscaled via external API")
            return Response(content=response.content, media_type="image/png")
        else:
            # Print error details and raise HTTP exception if the external API did not return 200
            print("Error response from external API:", response.json())
            raise HTTPException(status_code=response.status_code, detail=response.json())

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
