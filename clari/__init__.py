import warnings

# Suppress the PyTorch warning about linalg_svd fallback to CPU on MPS backend
warnings.filterwarnings("ignore", category=UserWarning, message=".*aten::linalg_svd.*MPS backend.*")
