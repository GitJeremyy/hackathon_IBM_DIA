# import lancedb
# from sentence_transformers import SentenceTransformer

# # --- Charger la base LanceDB et le modèle ---
# db = lancedb.connect("lancedb_questions")
# table = db.open_table("qa_table")
# # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# model = SentenceTransformer("intfloat/multilingual-e5-base")

# # --- Boucle interactive ---
# while True:
#     query = input("\n❓ Quelle est votre question ? (ou 'quit' pour sortir)\n> ")
#     if query.lower() in ["quit", "exit"]:
#         print("👋 Fin du programme.")
#         break

#     # Encoder la question et faire la recherche
#     query_vec = model.encode(query)
#     results = table.search(query_vec, vector_column_name="question_embedding").limit(3).to_pandas()

#     # Affichage des 3 résultats les plus pertinents
#     print("\n🔍 Résultats les plus pertinents :\n")
#     for i, row in results.iterrows():
#         print(f"🧠 Question similaire {i+1}: {row['question']}")
#         print(f"💬 Réponse : {row['answer']}")
#         print(f"🏫 École : {row['ecole']} | 🌍 Langue : {row['langue']}")
#         print("-" * 80)


import lancedb
from sentence_transformers import SentenceTransformer
import pandas as pd

# --- Connect to the LanceDB database and load the model ---
db = lancedb.connect("lancedb_questions")
table = db.open_table("qa_table")
model = SentenceTransformer("intfloat/multilingual-e5-base")


def search_question(question: str, school: str, language: str, top_k: int = 3):
    """
    Search for the most similar questions in LanceDB for a given school.

    Args:
        question (str): User question.
        school (str): School name (e.g., 'esilv', 'emlv').
        top_k (int): Number of results to return (default = 3).

    Returns:
        pd.DataFrame: The top_k matching question/answer pairs.
    """
    # Encode the query
    query_vec = model.encode(question)

    # Retrieve the 20 most similar questions
    results = table.search(query_vec, vector_column_name="question_embedding").limit(30).to_pandas()

    # Filter results by school (the column contains strings like "esilv,emlv")
    mask = results["ecole"].str.lower().str.contains(school.lower())
    filtered = results[mask].head(top_k)

    # Filter results by language (the column contains strings like "Français,Anglais")
    mask = results["langue"].str.lower().str.contains(language.lower())
    filtered = results[mask].head(top_k)

    if filtered.empty:
        print(f"❌ No results found for school '{school}'.")
        return None

    print(f"\n🔍 Top {top_k} relevant results for '{school}':\n")
    for i, row in filtered.iterrows():
        print(f"🧠 Similar Question {i+1}: {row['question']}")
        print(f"💬 Answer: {row['answer']}")
        print(f"🏫 School(s): {row['ecole']} | 🌍 Language: {row['langue']}")
        print("-" * 80)

    return filtered[['question', 'answer', 'ecole', 'langue']]


# --- Example usage ---
if __name__ == "__main__":
    q = "How many absences are allowed ?"
    s = "esilv"
    df_res = search_question(q, s)
