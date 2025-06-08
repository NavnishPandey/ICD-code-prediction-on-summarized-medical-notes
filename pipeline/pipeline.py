

from ray import serve
from fastapi import FastAPI, Request
from ray.util.actor_pool import ActorPool
from models.summarizer import Summarizer
from models.ICD_predictor import ICDPredictor

# Create FastAPI instance
api = FastAPI()

@serve.deployment
@serve.ingress(api)
class MedicalNotePipeline:
    def __init__(self, num_workers=1, use_gpu=True):
        # Assign half-GPU to each actor if use_gpu is True
        gpu_per_actor = 0.5 if use_gpu else 0

        summarizers = [Summarizer.options(num_gpus=gpu_per_actor).remote() for _ in range(num_workers)]
        predictors = [ICDPredictor.options(num_gpus=gpu_per_actor).remote() for _ in range(num_workers)]

        self.summarizer_pool = ActorPool(summarizers)
        self.predictor_pool = ActorPool(predictors)

    @api.post("/")
    async def process_dialogue(self, request: Request):
        payload = await request.json()
        raw_dialogue = payload.get("dialogue", "")
        print("📥 Received dialogue:", raw_dialogue)

        # Summary step
        summary_futures = self.summarizer_pool.map(lambda a, d: a.summarize.remote(d), [raw_dialogue])
        summary = next(summary_futures)
        print("📝 Summary:", summary)

        # ICD prediction step
        icd_futures = self.predictor_pool.map(lambda a, s: a.predict.remote(s), [summary])
        icd_code = next(icd_futures)
        print("🔢 ICD Code:", icd_code)

        return {
            "summary": summary,
            "icd_code": icd_code
        }
