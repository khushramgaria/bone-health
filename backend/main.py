from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
from models import predict_fracture, predict_bone_health_from_image, predict_bone_health_from_values
import json
import os


app = FastAPI(
    title="Bone Health AI API",
    description="Multi-ViT Enhanced CNN Framework for Bone Health Assessment",
    version="1.0.0"
)


# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/predict-fracture")
async def fracture_detection(file: UploadFile = File(...)):
    """
    Fracture Detection Module (X-ray based)
    """
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        result = predict_fracture(image)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/predict-bone-health/image")
async def bone_health_from_image(file: UploadFile = File(...)):
    """
    Bone Health Module - DEXA scan image upload
    """
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        result = predict_bone_health_from_image(image)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/predict-bone-health/values")
async def bone_health_from_values(
    bmd_value: float = Form(None),
    t_score: float = Form(None),
    z_score: float = Form(None),
    age: int = Form(None),
    gender: str = Form(None)
):
    """
    Bone Health Module - Manual BMD values input
    """
    try:
        result = predict_bone_health_from_values(
            bmd_value=bmd_value,
            t_score=t_score,
            z_score=z_score,
            age=age,
            gender=gender
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/api/metrics/all-data")
async def get_all_metrics_data():
    """
    Get ALL metrics data for Model Performance dashboard
    """
    try:
        all_data = {}
        
        # Summary
        with open("metrics_output/summary.json", "r") as f:
            all_data["summary"] = json.load(f)
        
        # Performance comparison
        with open("metrics_output/performance_comparison.json", "r") as f:
            all_data["performance_comparison"] = json.load(f)
        
        # Class-wise performance
        with open("metrics_output/classwise_performance.json", "r") as f:
            all_data["classwise_performance"] = json.load(f)
        
        # Dataset distribution
        with open("metrics_output/dataset_distribution.json", "r") as f:
            all_data["dataset_distribution"] = json.load(f)
        
        return JSONResponse(content=all_data)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



@app.get("/api/metrics/image/{image_name}")
async def get_metric_image(image_name: str):
    """
    Serve metric images
    """
    file_path = os.path.join("metrics_output", image_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return JSONResponse(content={"error": "Image not found"}, status_code=404)


@app.get("/api/metrics/data")
async def get_metrics_data():
    """
    Get all metrics data (JSON)
    """
    try:
        metrics_data = {}
        
        with open("metrics_output/metrics.json", "r") as f:
            metrics_data["metrics"] = json.load(f)
        
        with open("metrics_output/classification_report.json", "r") as f:
            metrics_data["classification_report"] = json.load(f)
        
        with open("metrics_output/per_class_accuracy.json", "r") as f:
            metrics_data["per_class_accuracy"] = json.load(f)
        
        return JSONResponse(content=metrics_data)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
def root():
    return {
        "message": "Bone Health AI Backend Running",
        "status": "operational",
        "endpoints": {
            "fracture_detection": "/api/predict-fracture",
            "bone_health_image": "/api/predict-bone-health/image",
            "bone_health_values": "/api/predict-bone-health/values",
            "model_performance": "/api/metrics/data"
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
