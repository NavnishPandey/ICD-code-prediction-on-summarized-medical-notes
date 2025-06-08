import requests

response = requests.post(
    "http://127.0.0.1:8000/MedicalNotePipeline",
    json={"dialogue": "Doctor: How are you? Patient: I feel dizzy and my head hurts."}
)

print("\n📝 Summary and ICD Prediction:")
print(response.json())
