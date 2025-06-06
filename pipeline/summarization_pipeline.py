import ray
#from utils.filters import DialogueFilter
from models.summarizer import Summarizer
from models.ICD_predictor import ICDPredictor


class MedicalNotePipeline:
    def __init__(self):
        ray.init(ignore_reinit_error=True)
        self.summarizer = Summarizer.options().remote()
        self.predictor = ICDPredictor.options().remote()

    def process_dialogue(self, raw_dialogue: str) -> dict:
        #cleaned = DialogueFilter.clean(raw_dialogue)
        print("summarization starting")
        summary_future = self.summarizer.summarize.remote(raw_dialogue)
        summary = ray.get(summary_future)
        ray.kill(self.summarizer)

        icd_future = self.predictor.predict.remote(summary)
        icd_code = ray.get(icd_future)
        ray.kill(self.predictor)

        return {
            "summary": summary,
            "icd_code": icd_code
        }
