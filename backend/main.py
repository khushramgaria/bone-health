from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
from models import predict_fracture, predict_bone_health


app = FastAPI(
    title="Bone Health AI API",
    description="Multi-ViT Enhanced CNN Framework for Bone Health Assessment",
    version="1.0.0"
)


# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/predict-fracture")
async def fracture_detection(file: UploadFile = File(...)):
    """
    Fracture Detection Module (X-ray based)
    Returns: prediction, confidence, explainability images
    """
    try:
        # Read uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Run prediction
        result = predict_fracture(image)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/predict-bone-health")
async def bone_health_prediction(file: UploadFile = File(...)):
    """
    Bone Health Module (DEXA scan based)
    Returns: BMD category, T-score, Z-score, risk level
    """
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        result = predict_bone_health(image)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
def root():
    return {
        "message": "Bone Health AI Backend Running",
        "status": "operational",
        "endpoints": {
            "fracture_detection": "/api/predict-fracture",
            "bone_health": "/api/predict-bone-health"
        }
    }


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
