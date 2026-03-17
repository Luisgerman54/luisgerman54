import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# Load model
booster = xgb.Booster()
booster.load_model("model.json")

# Remove feature names (important for converter)
booster.feature_names = None

# Detect real feature count
feature_count = booster.num_features()
print("Feature count:", feature_count)

# Define ONNX input
initial_type = [("input", FloatTensorType([None, feature_count]))]

# Convert
onnx_model = onnxmltools.convert_xgboost(
    booster,
    initial_types=initial_type
)

# Save
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX conversion complete")