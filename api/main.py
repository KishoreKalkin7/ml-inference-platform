from fastapi import FastAPI
from pydantic import BaseModel
import redis
import uuid
import json

app = FastAPI(title="ML Inference Platform")

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

class PredictRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictRequest):
    job_id = str(uuid.uuid4())
    payload = json.dumps({"job_id": job_id, "text": request.text})
    r.rpush("jobs", payload)
    return {"job_id": job_id, "status": "queued"}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    result = r.get(f"result:{job_id}")
    if result is None:
        return {"job_id": job_id, "status": "processing"}
    return {"job_id": job_id, "status": "done", "result": json.loads(result)}