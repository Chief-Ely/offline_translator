import sys
import json
import os

# Add the model path to system path
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'tagalog_to_cebuano')
sys.path.append(model_dir)

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class TokenizerService:
    def __init__(self):
        self.tokenizer = None
    
    def init_tokenizer(self, model_path):
        if not HAS_TRANSFORMERS:
            return json.dumps({"error": "transformers not available"})
        
        try:
            # Use the provided model path
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            return json.dumps({"status": "success", "vocab_size": len(self.tokenizer)})
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def tokenize(self, text):
        if not HAS_TRANSFORMERS:
            return json.dumps({"error": "transformers not available"})
        
        if self.tokenizer is None:
            return json.dumps({"error": "Tokenizer not initialized"})
        
        try:
            # Tokenize like in Colab
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            result = {
                "input_ids": inputs["input_ids"].tolist()[0],
                "attention_mask": inputs["attention_mask"].tolist()[0],
                "tokens": self.tokenizer.tokenize(text)
            }
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

# Global instance
service = TokenizerService()

if __name__ == "__main__":
    # Read command from stdin
    input_data = sys.stdin.read().strip()
    if not input_data:
        print(json.dumps({"error": "No input"}))
        sys.exit(1)
    
    try:
        data = json.loads(input_data)
        command = data.get("command", "")
        args = data.get("args", [])
        
        if command == "init":
            result = service.init_tokenizer(args[0] if args else "")
        elif command == "tokenize":
            result = service.tokenize(args[0] if args else "")
        else:
            result = json.dumps({"error": f"Unknown command: {command}"})
        
        print(result)
    except Exception as e:
        print(json.dumps({"error": str(e)}))