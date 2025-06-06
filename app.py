from pipeline.summarization_pipeline import MedicalNotePipeline
from huggingface_hub import login

token="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"
login(token)

example_dialogue = """
    Doctor: Good morning, how are you feeling today?
    Patient: I've been having chest pain for a few days, especially when I exert myself.
    Doctor: Can you describe the pain? Is it sharp or dull?
    Patient: It's a tight, squeezing feeling in my chest.
    Doctor: Any other symptoms, like shortness of breath or cough?
    Patient: No cough, but I do feel a bit short of breath sometimes.
    Doctor: Alright, we'll run some tests to check your heart and lungs.
    """
def main():
    print("==== Medical Dialogue Summarizer and ICD Predictor ====")
    #dialogue = input("Paste medical dialogue:\n")

    pipeline = MedicalNotePipeline()
    result = pipeline.process_dialogue(example_dialogue)
    
    print("\n--- Medical Note Summary ---\n")
    print(result["summary"])
    print("\n--- Predicted ICD Code ---\n")
    print(result["icd_code"])

if __name__ == "__main__":
    main()
