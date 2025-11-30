import os
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import numpy as np

# Adjust imports to be robust to how the script is run
if __package__ is None or __package__ == '':
    # Allows running the script directly for development `python app/main.py`
    from inference import DeepfakeDetector
    from utils_video import get_face_detector, sample_video_frames, extract_faces
else:
    # Allows running with uvicorn from root `uvicorn app.main:app`
    from .inference import DeepfakeDetector
    from .utils_video import get_face_detector, sample_video_frames, extract_faces

# --- Configuration ---
MODEL_NAME = "efficientnet_b3"
MODEL_PATH = f"models/best_model_{MODEL_NAME}.pth"
FRAMES_TO_SAMPLE = 30
FACE_CONF_THRESHOLD = 0.90
PREDICTION_THRESHOLD = 0.5  # If avg fake score > 0.5, classify as FAKE

# --- Global Objects ---
# These are initialized once when the application starts.
app = FastAPI(title="DeepSight Deepfake Detection API")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
try:
    face_detector = get_face_detector(DEVICE)
    deepfake_detector = DeepfakeDetector(model_path=MODEL_PATH, model_name=MODEL_NAME, device=DEVICE)
except Exception as e:
    print(f"Fatal error during model initialization: {e}")
    # In a real-world scenario, you might want the app to fail fast if models can't load.
    face_detector = None
    deepfake_detector = None


@app.on_event("startup")
async def startup_event():
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model file not found at {MODEL_PATH}. The /predict endpoint will not work.")
    if face_detector is None or deepfake_detector is None:
        print("WARNING: One or more models failed to initialize. The /predict endpoint will not work.")


@app.get("/")
async def read_root():
    """Simple root endpoint to check if the API is running."""
    return {"message": "DeepSight Deepfake Detection API is running.  Go to /docs for API documentation."}


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    """
    Accepts a video file, performs deepfake detection, and returns the result.
    """
    if deepfake_detector is None or face_detector is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please check server logs.")

    # Use a temporary file to save the uploaded video
    # On Windows, NamedTemporaryFile opens a file that can't be opened by another process
    # until it's closed. We need to write to it, close it, and then let our video
    # processing functions use the path. The `with` block ensures it's deleted afterward.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name
    
    try:

        # 1. Sample frames from the video
        frame_indices = sample_video_frames(temp_video_path, FRAMES_TO_SAMPLE)
        if not frame_indices:
            raise HTTPException(status_code=400, detail="Could not process video or sample frames.")

        # 2. Extract faces from the sampled frames
        faces = extract_faces(temp_video_path, frame_indices, face_detector, FACE_CONF_THRESHOLD)
        if not faces:
            return JSONResponse(content={
                "verdict": "UNCERTAIN",
                "reason": "No faces were detected with sufficient confidence.",
                "faces_detected": 0,
                "average_fake_probability": 0.0
            })

        # 3. Get predictions for each face
        face_predictions = deepfake_detector.predict_faces(faces)

        # 4. Aggregate results and make a final decision
        average_fake_prob = np.mean(face_predictions)
        final_verdict = "FAKE" if average_fake_prob > PREDICTION_THRESHOLD else "REAL"

        return JSONResponse(content={
            "verdict": final_verdict,
            "faces_detected": len(faces),
            "average_fake_probability": round(average_fake_prob, 4),
            "prediction_threshold": PREDICTION_THRESHOLD
        })
    finally:
        os.remove(temp_video_path) # Clean up the temporary file


if __name__ == "__main__":
    print("--- Starting DeepSight API in development mode ---")
    print(f"Model Path: {os.path.abspath(MODEL_PATH)}")
    uvicorn.run(app, host="127.0.0.1", port=8000)