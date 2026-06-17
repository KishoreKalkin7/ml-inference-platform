from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import os

print("Converting DistilBERT to ONNX format...")
print("This will take 2-3 minutes...")

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
save_path = "./models/onnx_model"

os.makedirs(save_path, exist_ok=True)

model = ORTModelForSequenceClassification.from_pretrained(
    model_name,
    export=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model converted and saved to {save_path}")
print("Testing ONNX model speed...")

import time

test_text = "This is absolutely amazing!"

start = time.time()
from transformers import pipeline
onnx_model = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)
result = onnx_model(test_text)
end = time.time()

print(f"ONNX result: {result}")
print(f"ONNX inference time: {round((end-start)*1000, 2)}ms")
print("Conversion complete!")