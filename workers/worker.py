import redis
import json
import time
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

r = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
    socket_timeout=30,
    socket_connect_timeout=30,
    retry_on_timeout=True
)

print("Loading ONNX optimised model...")
model_path = "./models/onnx_model"
model = ORTModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
onnx_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print("ONNX model loaded! Worker ready for jobs...")

while True:
    try:
        job = r.blpop("jobs", timeout=5)

        if job is None:
            print("Waiting for jobs...")
            continue

        payload = json.loads(job[1])
        job_id = payload["job_id"]
        text = payload["text"]

        print(f"Processing job {job_id}: {text}")

        start_time = time.time()
        prediction = onnx_pipeline(text)[0]
        end_time = time.time()

        latency_ms = round((end_time - start_time) * 1000, 2)

        result = {
            "prediction": prediction["label"],
            "confidence": round(prediction["score"], 4),
            "latency_ms": latency_ms,
            "model": "onnx_optimised"
        }

        r.set(f"result:{job_id}", json.dumps(result))
        print(f"Done — {result}")

    except Exception as e:
        print(f"Error: {e} — retrying in 3 seconds...")
        time.sleep(3)