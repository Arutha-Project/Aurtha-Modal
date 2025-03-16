from pathlib import Path
import tempfile
import shutil
from inference import SignLanguageInference

class SignLanguageService:
    def __init__(self, model_path: str):
        self.inference = SignLanguageInference(model_path)

    def predict_from_file(self, file) -> str:
        """Save the uploaded file temporarily, run prediction, and return result."""
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = temp.name
            shutil.copyfileobj(file, temp)

        try:
            return self.inference.predict(temp_path)
        finally:
            # Clean up the temporary file
            Path(temp_path).unlink(missing_ok=True)
