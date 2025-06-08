from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ray

@ray.remote
class ICDPredictor:
    def __init__(self, model_name="microsoft/phi-2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

        self.prompt_template = """
You are a clinical coding assistant. Based on the conversation summary below, return only the most relevant ICD-10 diagnosis code(s) from this list.

### ICD-10 Code List:
- A00: Cholera  
  > Symptoms: severe watery diarrhea, dehydration, vomiting  
- B20: HIV disease  
  > Symptoms: immunodeficiency, recurrent infections, weight loss  
- C34: Malignant neoplasm of bronchus and lung  
  > Symptoms: persistent cough, hemoptysis, chest pain, weight loss  
- D50: Iron deficiency anemia  
  > Symptoms: fatigue, pallor, weakness, shortness of breath  
- E11: Type 2 diabetes mellitus  
  > Symptoms: high blood sugar, frequent urination, increased thirst, blurred vision  
- F32: Major depressive disorder  
  > Symptoms: sadness, lack of interest, fatigue, sleep/appetite changes  
- G40: Epilepsy  
  > Symptoms: seizures, loss of consciousness, aura  
- H52: Disorders of refraction and accommodation  
  > Symptoms: blurred vision, myopia, hyperopia  
- I10: Essential (primary) hypertension  
  > Symptoms: high blood pressure, headaches, dizziness, often asymptomatic  
- J45: Asthma  
  > Symptoms: wheezing, shortness of breath, coughing, chest tightness  
- K35: Acute appendicitis  
  > Symptoms: abdominal pain (lower right), fever, nausea  
- M54: Dorsalgia (Back Pain)  
  > Symptoms: upper/lower back pain, stiffness, limited movement  
- N39: Urinary tract infection  
  > Symptoms: burning urination, frequent urination, lower abdominal pain  
- R51: Headache  
  > Symptoms: pain in head or neck region, tension or migraine-like  
- Z00: General examination  
  > Used when no complaint or diagnosis is present (routine checkup) 

### Medical Summary:
{summary}

### Your Response must be just ICD Codes Only:
"""

    def predict(self, summary: str) -> str:
        prompt = self.prompt_template.format(summary=summary)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        icd_output = response.split("ICD Codes Only:")[-1].strip()
        return icd_output