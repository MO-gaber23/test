from skimage.feature import hog, local_binary_pattern
from typing import Tuple, List, Any
from src.constant.constant import *
import onnxruntime as ort
import numpy as np
import cv2


class NeuroSpiralPredictor:
    """
        A predictor class that uses an ONNX model to classify images based on
        pre-extracted HOG and LBP features combined with raw image data.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initializes the NeuroSpiralPredictor by loading the ONNX model.

        Args:
            model_path (str): The file path to the ONNX model.

        Raises:
            RuntimeError: If the model fails to load.
        """
        try:
            self.session: ort.InferenceSession = ort.InferenceSession(model_path, providers=providers)
            self.input_names: List[str] = [i.name for i in self.session.get_inputs()]
            self.output_name: str = self.session.get_outputs()[0].name
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _extract_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extracts HOG and LBP features from the input image.

        Args:
            img (np.ndarray): The input image in BGR or grayscale format.

        Returns:
            np.ndarray: A concatenated 1D array of HOG and LBP features.
        """
        if len(img.shape) == 3:
            gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        img_resized: np.ndarray = cv2.resize(gray, (224, 224))

        hog_features: np.ndarray = hog(
            img_resized,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True
        )

        lbp: np.ndarray = local_binary_pattern(img_resized, radius, n_points, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
        lbp_hist = lbp_hist.astype("float32")
        lbp_hist /= (lbp_hist.sum() + 1e-6)

        return np.concatenate([hog_features, lbp_hist]).astype(np.float32)

    def predict(self, img: np.ndarray) -> Tuple[str, float]:
        """
        Performs inference on the provided image.

        Args:
            img (np.ndarray): The input image.

        Returns:
            Tuple[str, float]: A tuple containing the classification label
                               ('PD' or 'HC') and the calculated probability.
        """
        img_input: np.ndarray = cv2.resize(img, (224, 224))

        if len(img_input.shape) == 2:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)

        img_tensor: np.ndarray = img_input.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        img_tensor = img_tensor / 255.0

        features: np.ndarray = self._extract_features(img)
        features = np.expand_dims(features, axis=0).astype(np.float32)

        inputs: Dict[str, np.ndarray] = {
            self.input_names[0]: img_tensor,
            self.input_names[1]: features
        }

        logits: float = float(self.session.run([self.output_name], inputs)[0].item())

        probability: float = 1 / (1 + np.exp(-logits))

        label: str = "PD" if probability < 0.5 else "HC"

        return label, round(probability, 4)





