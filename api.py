from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
import io
import matplotlib.pyplot as plt
import mediapipe as mp
from main import detectHandsLandmarks, countFingers

app = FastAPI(title="Finger Counting API")

# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

@app.post("/analyze/")
async def analyze_image(image: UploadFile):
    try:
        # Read the uploaded image
        image_bytes = await image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform hands landmarks detection on the image
        _, results = detectHandsLandmarks(frame, hands, display=False)

        # Count fingers on the detected hands
        _, _, count = countFingers(frame, results, display=False)

        # Return the finger counts
        return count
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
