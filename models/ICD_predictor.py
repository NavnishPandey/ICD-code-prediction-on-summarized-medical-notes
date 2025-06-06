import ray
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain.prompts import PromptTemplate

@ray.remote
class ICDPredictor:
    def __init__(self, model_name="microsoft/phi-2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.prompt_template = PromptTemplate(
            input_variables=["summary"],
            template="""
You are a clinical coding assistant.

Given the following medical summary, identify the most appropriate ICD-10 diagnosis code(s) from the list below. Respond only with the most relevant ICD-10 code(s).

### ICD-10 Code List (Shortened):
- A00: Cholera  
- B20: HIV disease  
- C34: Malignant neoplasm of bronchus and lung  
- D50: Iron deficiency anemia  
- E11: Type 2 diabetes mellitus  
- F32: Major depressive disorder  
- G40: Epilepsy  
- H52: Disorders of refraction and accommodation  
- I10: Essential (primary) hypertension  
- J45: Asthma  
- K35: Acute appendicitis  
- M54: Dorsalgia (Back Pain)  
- N39: Urinary tract infection  
- R51: Headache  
- Z00: General examination

### Medical Summary:
{summary}

Your Response (ICD-10 code only):
"""
        )

    def predict(self, summary: str) -> str:
        # Simulate prompt (not used directly in classification, but could be for future generation)
        prompt = self.prompt_template.format(summary=summary)

        inputs = self.tokenizer(summary, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()

        return f"ICD-Code-{predicted_label}"  # Optional: map label index to actual code if known
