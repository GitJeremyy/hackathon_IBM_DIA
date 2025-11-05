from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai import Credentials
import json


# --- Identifiants ---
api_key = "Yv4DvZ5xubJXzmy7ywF37VBxsUkTpmch7iaLgRG60TOF"
project_id = "1b291e34-ae01-4e65-bcb7-46426fee83e1"
region = "eu-de"  # ou "us-south" selon ta région

import json
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials


creds = Credentials(
    url=f"https://{region}.ml.cloud.ibm.com",
    api_key=api_key
)

# --- Modèle ---
model = ModelInference(
    model_id="meta-llama/llama-3-3-70b-instruct",
    credentials=creds,
    project_id=project_id
)

# --- Prompt ---
prompt = (
    "Explique-moi clairement et en plusieurs phrases le principe du "
    "Retrieval-Augmented Generation (RAG) en intelligence artificielle, "
    "avec un exemple d'application concrète."
)

# --- Appel du modèle avec paramètres de génération ---
params = {
    "max_new_tokens": 500,
    "temperature": 0.7,
    "repetition_penalty": 1.1,
    "stop_sequences": []
}

response = model.generate(prompt=prompt, params=params)

# --- Afficher le texte généré ---
print(response['results'][0]['generated_text'])

# --- Sauvegarder la réponse complète en JSON ---
with open("response.json", "w", encoding="utf-8") as f:
    json.dump(response, f, ensure_ascii=False, indent=4)

print("✅ Réponse enregistrée dans response.json")