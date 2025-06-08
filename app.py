import ray
from ray import serve
import time, os
from dotenv import load_dotenv
from pipeline.pipeline import MedicalNotePipeline  
from huggingface_hub import login

load_dotenv()

token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(token)

ray.init(ignore_reinit_error=True)
print(ray.available_resources())

serve.run(MedicalNotePipeline.bind(), route_prefix="/MedicalNotePipeline")

print("🚀 Service running at http://127.0.0.1:8000/MedicalNotePipeline")
print("Press Ctrl+C to stop.")

# Keep alive
while True:
    time.sleep(10)