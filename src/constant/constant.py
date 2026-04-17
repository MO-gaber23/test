from pathlib import Path

# ======================================================
# Dataset & Paths Configuration
# ======================================================
base_dir = Path(__file__).resolve().parent.parent.parent
model_path = base_dir / "checkpoint" / "best_model.onnx"



# ======================================================
# ONNX Runtime Providers
# ======================================================
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]


# ======================================================
# LBP settings (must match training)
# ======================================================
radius = 1
n_points = 8 * radius