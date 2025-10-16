import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class Translator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.encoder_sess = None
        self.decoder_sess = None
        self.tokenizer = None

    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        encoder_path = os.path.join(self.model_path, "encoder_model.onnx")
        decoder_path = os.path.join(self.model_path, "decoder_model.onnx")
        self.encoder_sess = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        self.decoder_sess = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])

    def translate(self, text: str, max_length: int = 64) -> str:
        inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        encoder_outputs = self.encoder_sess.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )

        decoder_input_ids = np.array([[self.tokenizer.pad_token_id]], dtype=np.int64)
        decoded_tokens = []

        for _ in range(max_length):
            decoder_outputs = self.decoder_sess.run(
                None,
                {
                    "input_ids": decoder_input_ids,
                    "encoder_hidden_states": encoder_outputs[0],
                    "encoder_attention_mask": attention_mask,
                },
            )
            next_token = int(np.argmax(decoder_outputs[0][0, -1]))
            if next_token == self.tokenizer.eos_token_id:
                break
            decoded_tokens.append(next_token)
            decoder_input_ids = np.hstack([decoder_input_ids, [[next_token]]])

        return self.tokenizer.decode(decoded_tokens, skip_special_tokens=True)

# Singleton
translator_instance = None

def init_translator(model_path: str):
    global translator_instance
    if translator_instance is None:
        translator_instance = Translator(model_path)
        translator_instance.init()

def translate(text: str) -> str:
    global translator_instance
    if translator_instance is None:
        raise RuntimeError("Translator not initialized. Call init_translator first.")
    return translator_instance.translate(text)
