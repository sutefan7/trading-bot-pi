"""
Model serving utilities for ONNX inference on the Pi.

Provides a simple `ModelPredictor` around ONNX Runtime with:
- Artifact loading (onnx model, scaler, thresholds, featureset, metadata)
- Shape and feature-count validation against the shared feature schema
- Convenience helpers for single/batch prediction and metadata access

⚠️ CRITICAL FIX: predict_one now returns Dict instead of float for compatibility
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import joblib
import numpy as np
import yaml

from loguru import logger

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except Exception:
    # Import is deferred to load time so importing this module won't fail
    ONNX_AVAILABLE = False

# Use the shared schema to ensure train/serve parity
try:
    from features.schema import get_feature_names
except ImportError:
    # Fallback if schema not found
    def get_feature_names() -> List[str]:
        logger.warning("features.schema not found, using default feature names")
        return []


@dataclass
class ModelBundle:
    """Container for a loaded ONNX model and its metadata."""
    session: Any
    feature_names: List[str]
    scaler: Any
    thresholds: Dict[str, Any]
    metadata: Dict[str, Any]
    version: str
    input_name: str
    output_name: str


class ModelPredictor:
    """High-level ONNX predictor with artifact loading and validation."""

    def __init__(self) -> None:
        self.bundle: Optional[ModelBundle] = None
        self.model_available: bool = False

    def load_artifact(self, artifact_dir: str) -> ModelBundle:
        """Load an exported artifact directory into a ModelBundle.

        Expected files (best-effort):
        - model.onnx (or any *.onnx)
        - scaler.pkl (optional)
        - thresholds.yaml (optional)
        - featureset.json (optional)
        - metadata.json (optional)
        """
        artifact_path = Path(artifact_dir)
        if not artifact_path.exists():
            raise ValueError("Artifact directory does not exist")

        # Locate ONNX model file
        onnx_path = artifact_path / "model.onnx"
        if not onnx_path.exists():
            candidates = list(artifact_path.glob("*.onnx"))
            if not candidates:
                raise ValueError("ONNX model not found")
            onnx_path = candidates[0]

        # Optional scaler
        scaler_path = artifact_path / "scaler.pkl"
        scaler = None
        if scaler_path.exists():
            try:
                scaler = joblib.load(scaler_path)
            except Exception as e:
                logger.warning(f"Failed to load scaler: {e}")

        # Optional thresholds
        thresholds: Dict[str, Any] = {}
        thresholds_path = artifact_path / "thresholds.yaml"
        if thresholds_path.exists():
            try:
                with open(thresholds_path, "r") as f:
                    thresholds = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load thresholds: {e}")

        # Feature names from featureset.json or schema fallback
        feature_names: List[str]
        featureset_path = artifact_path / "featureset.json"
        if featureset_path.exists():
            try:
                with open(featureset_path, "r") as f:
                    featureset = json.load(f)
                names = featureset.get("feature_names")
                feature_names = list(names) if names else get_feature_names()
            except Exception as e:
                logger.warning(f"Failed to load featureset: {e}")
                feature_names = get_feature_names()
        else:
            feature_names = get_feature_names()

        # Optional metadata
        metadata: Dict[str, Any] = {}
        metadata_path = artifact_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")

        version = str(metadata.get("model_version") or metadata.get("version") or "unknown")

        # Create ONNX Runtime session (will raise on invalid file, as tests expect)
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX runtime not available")

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

        input_name = session.get_inputs()[0].name if session.get_inputs() else "input"
        output_name = session.get_outputs()[0].name if session.get_outputs() else "output"

        bundle = ModelBundle(
            session=session,
            feature_names=feature_names,
            scaler=scaler,
            thresholds=thresholds,
            metadata=metadata,
            version=version,
            input_name=input_name,
            output_name=output_name,
        )

        self.bundle = bundle
        self.model_available = True
        return bundle

    def predict_one(self, bundle: Optional[ModelBundle], features: np.ndarray) -> Dict[str, Any]:
        """
        Predict a single sample. Validates shape and feature count.
        
        ⚠️ CRITICAL FIX: Now returns Dict instead of float!
        
        Args:
            bundle: ModelBundle with loaded model
            features: 1D numpy array of features
            
        Returns:
            Dictionary with prediction results:
            {
                'prediction': float,  # Raw prediction
                'buy': bool,          # Buy signal
                'sell': bool,         # Sell signal
                'hold': bool,         # Hold signal
                'buy_prob': float,    # Buy probability
                'sell_prob': float,   # Sell probability
                'proba': float,       # Max probability
                'confidence': float,  # Confidence score
                'model_version': str  # Model version
            }
        """
        active_bundle = bundle or self.bundle
        if active_bundle is None:
            raise ValueError("Model not available")

        if features.ndim != 1:
            raise ValueError("Features must be 1D array")

        expected = len(active_bundle.feature_names)
        if features.shape[0] != expected:
            raise ValueError(f"Feature count mismatch: expected {expected}, got {features.shape[0]}")

        X = features.reshape(1, -1).astype(np.float32)
        if active_bundle.scaler is not None:
            try:
                X = active_bundle.scaler.transform(X)
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}")

        if active_bundle.session is None:
            raise ValueError("Model not available")

        outputs = active_bundle.session.run(None, {active_bundle.input_name: X})
        raw_prediction = float(outputs[0].flatten()[0]) if outputs else 0.5
        
        # Get thresholds
        buy_threshold = active_bundle.thresholds.get('buy_threshold', 0.6)
        sell_threshold = active_bundle.thresholds.get('sell_threshold', 0.4)
        
        # Interpret as probabilities (assuming binary classifier or regression [0,1])
        # For binary: >0.6 = buy, <0.4 = sell, else hold
        buy_signal = raw_prediction > buy_threshold
        sell_signal = raw_prediction < sell_threshold
        hold_signal = not (buy_signal or sell_signal)
        
        # Calculate probabilities
        # Assuming model outputs buy probability (0 to 1)
        buy_prob = raw_prediction
        sell_prob = 1.0 - raw_prediction
        
        # Confidence is distance from neutral (0.5)
        confidence = abs(raw_prediction - 0.5) * 2.0  # 0 to 1
        
        return {
            'prediction': raw_prediction,
            'buy': buy_signal,
            'sell': sell_signal,
            'hold': hold_signal,
            'buy_prob': buy_prob,
            'sell_prob': sell_prob,
            'proba': max(buy_prob, sell_prob),
            'confidence': confidence,
            'model_version': active_bundle.version
        }

    def predict_batch(self, bundle: Optional[ModelBundle], features: np.ndarray) -> np.ndarray:
        """Predict a batch of samples. Requires 2D features array."""
        active_bundle = bundle or self.bundle
        if active_bundle is None:
            raise ValueError("Model not available")

        if features.ndim != 2:
            raise ValueError("Features must be 2D array")

        if features.shape[1] != len(active_bundle.feature_names):
            raise ValueError("Feature count mismatch")

        X = features.astype(np.float32)
        if active_bundle.scaler is not None:
            try:
                X = active_bundle.scaler.transform(X)
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}")

        if active_bundle.session is None:
            raise ValueError("Model not available")

        outputs = active_bundle.session.run(None, {active_bundle.input_name: X})
        return np.asarray(outputs[0]).flatten() if outputs else np.array([], dtype=np.float32)

    def get_model_version(self, bundle: Optional[ModelBundle]) -> str:
        active_bundle = bundle or self.bundle
        if active_bundle is None:
            return "unknown"
        return str(active_bundle.version or active_bundle.metadata.get("model_version", "unknown"))

    def get_model_info(self, bundle: Optional[ModelBundle]) -> Dict[str, Any]:
        active_bundle = bundle or self.bundle
        if active_bundle is None:
            return {"error": "No model loaded"}
        return {
            "version": self.get_model_version(active_bundle),
            "feature_count": len(active_bundle.feature_names),
            "thresholds": active_bundle.thresholds or {},
            "metadata": active_bundle.metadata or {},
        }

    def health_check(self, bundle: Optional[ModelBundle]) -> Dict[str, Any]:
        active_bundle = bundle or self.bundle
        if active_bundle is None:
            return {"status": "error", "message": "No model loaded"}
        if active_bundle.session is None:
            return {"status": "error", "message": "No ONNX session"}
        try:
            _ = active_bundle.input_name  # Accessing properties should succeed
            return {
                "status": "healthy",
                "feature_count": len(active_bundle.feature_names),
                "version": self.get_model_version(active_bundle),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Convenience functions expected by tests
def load_artifact(artifact_dir: str) -> ModelBundle:
    predictor = ModelPredictor()
    return predictor.load_artifact(artifact_dir)


def predict_one(bundle: Optional[ModelBundle], features: np.ndarray) -> Dict[str, Any]:
    """⚠️ CRITICAL FIX: Now returns Dict instead of float"""
    predictor = ModelPredictor()
    return predictor.predict_one(bundle, features)


def get_model_version(bundle: Optional[ModelBundle]) -> str:
    predictor = ModelPredictor()
    return predictor.get_model_version(bundle)



