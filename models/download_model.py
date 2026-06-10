from transformers import pipeline
import os

print("Downloading DistilBERT sentiment model...")
print("This will take 2-3 minutes on first run...")

model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

test = model("This movie was absolutely amazing!")
print(f"Model working! Test result: {test}")
print("Model downloaded and cached successfully!")