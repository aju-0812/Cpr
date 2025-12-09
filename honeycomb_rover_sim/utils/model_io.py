import joblib
import os
from honeycomb_rover_sim.utils.logger import logger

def save_model(obj, filepath):
    """Save an object to a file using joblib."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(obj, filepath)
        logger.info(f"Model saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return False

def load_model(filepath):
    """Load an object from a file using joblib."""
    try:
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return None
        obj = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
