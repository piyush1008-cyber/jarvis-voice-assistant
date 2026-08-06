import os
import cv2
import logging

# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

def detect_emotion(frame):
    """
    Analyzes the given image frame and returns the dominant emotion.
    Returns:
        dominant_emotion (str): Capitalized dominant emotion (e.g., 'Happy', 'Neutral').
        full_result (dict): Complete raw analysis output or error details.
    """
    if not DEEPFACE_AVAILABLE:
        return "Neutral", {"dominant_emotion": "Neutral", "message": "DeepFace not installed"}

    try:
        # Analyze the frame for emotion. 
        # enforce_detection=False prevents crash if face is partially visible or absent.
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # DeepFace returns a list of results if multiple faces or versions; take the first one
        if isinstance(result, list):
            result = result[0]
            
        dominant = result.get('dominant_emotion', 'neutral')
        return dominant.capitalize(), result
    except Exception as e:
        return "Neutral", {"dominant_emotion": "neutral", "error": str(e)}
