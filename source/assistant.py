import json
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials
from source.search_question import search_question  # ⚠️ Ton fichier précédent
from sentence_transformers import SentenceTransformer
from langdetect import detect, detect_langs


# --- IBM Watsonx credentials ---
API_KEY = "-4"
PROJECT_ID = "21e4c9cf-b356-4071-9b48-e5aeb3aa889f"
REGION = "eu-de"

# --- Initialize credentials and model ---
creds = Credentials(
    url=f"https://{REGION}.ml.cloud.ibm.com",
    api_key=API_KEY
)

llm_model = ModelInference(
    model_id="meta-llama/llama-3-3-70b-instruct",
    credentials=creds,
    project_id=PROJECT_ID
)


def school_assistant(question: str, school: str):
    """
    Use retrieved QA context + Llama-3 to answer a student's question.
    If not enough context, redirect to a contact form.
    """

    # Step 0 - Retrieve language
    language = detect(question)
    if language=="fr":
        language="Français"
    else:
        language="English"
    # Step 1️⃣ - Retrieve relevant Q&A from LanceDB
    retrieved = search_question(question, school, language)
    if retrieved is None or retrieved.empty:
        if(language=="Français"):
            print(f"❌ Aucun contexte trouvé pour '{school}', redirection vers un formulaire.")
            return "Je suis désolé, je n'ai pas pu trouver la réponse à votre question. s'il vous plaît utilisez le formulaire suivant: https://forms.office.com/"
        else:    
            print(f"❌ No context found for '{school}', redirecting to form.")
            return "I'm sorry, I couldn't find relevant information. Please use the contact form: https://forms.office.com/"

    # Step 2️⃣ - Build the context string from the retrieved QA
    context = ""
    for _, row in retrieved.iterrows():
        context += f"Q: {row['question']}\nA: {row['answer']}\n\n"

    # Step 3️⃣ - Build the prompt for the LLM
    if language=="Français":
        prompt = (
            f"Vous êtes un assistant administratif de l'école '{school.upper()}'. "
            f"Répondez à la question suivante de l'étudiant aussi clairement et utilement que possible, "
            f"en utilisant UNIQUEMENT les informations fournies dans le contexte de questions-réponses ci-dessous. "
            f"Si le contexte ne contient pas suffisamment d'informations pour répondre, répondez : "
            f"Je suis désolé, je n'ai pas trouvé la réponse. Veuillez utiliser le formulaire de contact : https://forms.office.com/ \n\n"
            f"Répondez dans la langue de la question."
            f"--- Contexte ---\n{context}\n"
            f"--- Question de l'étudiant ---\n{question}\n\n"
            f"--- Votre réponse utile ---"
            f"Donnez une réponse simple, unique et courte."
        )
    else:
        prompt = (
            f"You are an administrative assistant for the school '{school.upper()}'. "
            f"Answer the following student's question as clearly and helpfully as possible, "
            f"using ONLY the information provided in the following Q&A context. "
            f"If the context does not contain enough information to answer, respond with: "
            f"I'm sorry, I couldn't find the answer. Please use the contact form: https://forms.office.com/ \n\n"
            f"Answer in the question's langage"
            f"--- Context ---\n{context}\n"
            f"--- Student question ---\n{question}\n\n"
            f"--- Your helpful answer ---"
            f"give a simple, unique and a short answer"
            
        )

    # Step 4️⃣ - Query the LLM
    params = {
        "max_new_tokens": 200,
        "temperature": 0.6,
        "repetition_penalty": 1.1
    }

    response = llm_model.generate(prompt=prompt, params=params)
    answer = response["results"][0]["generated_text"].strip()

    # Step 5️⃣ - Display and save result
    if(language=="Français"):
        print(f"\n🤖 Réponse de l'assistant::\n{answer}\n")
    else:
        print(f"\n🤖 Assistant response:\n{answer}\n")

    with open("assistant_response.json", "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=4)

    print("✅ Full JSON response saved to assistant_response.json")
    return answer


# --- Example usage ---
if __name__ == "__main__":
    q = "A combien d'absence ai-je droit?"
    s = "esilv"
    print(school_assistant(q, s))
