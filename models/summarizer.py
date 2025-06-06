import ray
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.prompts import PromptTemplate

@ray.remote
class Summarizer:
    def __init__(self, model_name="google/gemma-2-2b-it"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.prompt = PromptTemplate(
            input_variables=["dialogue"],
            template="Summarize the following doctor-patient conversation into medical notes:\n\n{dialogue}"
        )

    def summarize(self, dialogue: str, max_length=256) -> str:
        prompt_text = self.prompt.format(dialogue=dialogue)
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True)
        summary_ids = self.model.generate(inputs["input_ids"], max_length=max_length, num_beams=4)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
