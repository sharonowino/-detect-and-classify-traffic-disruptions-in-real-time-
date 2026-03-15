fast_api.py# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# --- App setup ---
app = FastAPI(
    title="ML Prediction API",
    description="Run binary or multi-class predictions using your trained models",
    version="1.0"
)

# ----------------------------------------------------------------
# --- Load models, scaler, and imputer at startup ---
# ----------------------------------------------------------------
MODELS_FOLDER = "my_api/models"

# Preprocessing tools
scaler = None
imputer = None

# Models separated by type
binary_models = {}
multi_models = {}

for filename in os.listdir(MODELS_FOLDER):
    if not filename.endswith(".pkl"):
        continue

    filepath = os.path.join(MODELS_FOLDER, filename)
    
    # Skip models that fail to load due to numpy version issues
    try:
        # Load scaler
        if filename.startswith("scaler"):
            scaler = joblib.load(filepath)
            print(f"[OK] Loaded scaler: {filename}")

        # Load imputer
        elif filename.startswith("imputer"):
            imputer = joblib.load(filepath)
            print(f"[OK] Loaded imputer: {filename}")

        # Load binary models
        elif "binary" in filename:
            model_name = filename.split("_binary_")[0]
            binary_models[model_name] = joblib.load(filepath)
            print(f"[OK] Loaded binary model: {model_name}")

        # Load multi models
        elif "multi" in filename:
            model_name = filename.split("_multi_")[0]
            multi_models[model_name] = joblib.load(filepath)
            print(f"[OK] Loaded multi model: {model_name}")
    except Exception as e:
        print(f"[WARN] Skipped {filename}: {e}")
        continue

print(f"\n[Ready!]")
print(f"   Binary models : {list(binary_models.keys())}")
print(f"   Multi models  : {list(multi_models.keys())}")
print(f"   Scaler loaded : {scaler is not None}")
print(f"   Imputer loaded: {imputer is not None}")


# ----------------------------------------------------------------
# --- Input schema ---
# ----------------------------------------------------------------
class PredictRequest(BaseModel):
    model_name: str        # e.g. "XGBoost", "RandomForest", "NeuralNet", "BEST"
    task: str              # "binary" or "multi"
    features: list[float]  # your list of numbers e.g. [23.5, 1.0, 4500.0, ...]

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "XGBoost",
                "task": "binary",
                "features": [23.5, 1.0, 4500.0, 0.8, 3.2]
            }
        }


# ----------------------------------------------------------------
# --- Helper: preprocess input ---
# ----------------------------------------------------------------
def preprocess(features: list[float]) -> np.ndarray:
    X = np.array(features).reshape(1, -1)

    # Step 1: Impute missing values if imputer is available
    if imputer is not None:
        X = imputer.transform(X)

    # Step 2: Scale features if scaler is available
    if scaler is not None:
        X = scaler.transform(X)

    return X


# ----------------------------------------------------------------
# --- Helper: run prediction ---
# ----------------------------------------------------------------
def run_prediction(model, X: np.ndarray):
    prediction = model.predict(X)

    # Get confidence/probability if model supports it
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        confidence = round(float(proba.max()), 4)
    else:
        confidence = None

    return prediction[0], confidence


# ----------------------------------------------------------------
# --- Endpoints ---
# ----------------------------------------------------------------

# Health check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "binary_models": list(binary_models.keys()),
        "multi_models": list(multi_models.keys()),
        "scaler_loaded": scaler is not None,
        "imputer_loaded": imputer is not None
    }

# Alternative health endpoint (for dashboard compatibility)
@app.get("/")
def root():
    return {
        "service": "ML Prediction API",
        "version": "1.0",
        "status": "ok",
        "endpoints": [
            "/health - Health check",
            "/models - List available models",
            "/predict - Make prediction"
        ]
    }

# List all available models
@app.get("/models")
def list_models():
    return {
        "binary_models": list(binary_models.keys()),
        "multi_models": list(multi_models.keys()),
        "usage_tip": "Pass model_name + task ('binary' or 'multi') in /predict"
    }

# Main prediction endpoint
@app.post("/predict")
def predict(request: PredictRequest):

    # --- Validate task ---
    if request.task not in ("binary", "multi"):
        raise HTTPException(
            status_code=400,
            detail="task must be 'binary' or 'multi'"
        )

    # --- Pick the right model ---
    model_pool = binary_models if request.task == "binary" else multi_models

    if request.model_name not in model_pool:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_name}' not found for task '{request.task}'. "
                   f"Available: {list(model_pool.keys())}"
        )

    # --- Validate features ---
    if not request.features:
        raise HTTPException(status_code=400, detail="features list cannot be empty")

    # --- Preprocess ---
    try:
        X = preprocess(request.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

    # --- Predict ---
    try:
        prediction, confidence = run_prediction(model_pool[request.model_name], X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # --- Return result ---
    return {
        "model_used": request.model_name,
        "task": request.task,
        "prediction": str(prediction),
        "confidence": confidence if confidence is not None else "N/A",
        "input_features": request.features
    }
