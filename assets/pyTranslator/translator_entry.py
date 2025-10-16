import sys
from translator import init_translator, translate

# Entry point for SeriousPython.run
# sys.argv[1] → command ("init" or "translate")
# sys.argv[2] → argument (model path or text)
if len(sys.argv) < 3:
    sys.exit(1)

command = sys.argv[1]

if command == "init":
    model_path = sys.argv[2]
    init_translator(model_path)
elif command == "translate":
    text = sys.argv[2]
    result = translate(text)
    print(result)
