import os
import sys
from translator import init_translator, translate

# Read environment variables
command = os.environ.get("COMMAND")       # "init" or "translate"
model_path = os.environ.get("MODEL_PATH") # path to your ONNX models
site_packages = os.environ.get("SERIOUS_PYTHON_SITE_PACKAGES")

# Optional: add site-packages to sys.path
if site_packages:
    sys.path.insert(0, site_packages)

# Execute command
if command == "init" and model_path:
    init_translator(model_path)
elif command == "translate":
    text = os.environ.get("TEXT")  # the text to translate
    if text:
        result = translate(text)
        print(result)
else:
    sys.exit(1)
