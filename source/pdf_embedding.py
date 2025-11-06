# --- Dépendances à installer si besoin ---
# pip install pypdf lancedb sentence-transformers pandas

import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
import lancedb
from sentence_transformers import SentenceTransformer

# -------- 1) Chargement du PDF --------
pdf_path = "/content/reglint.pdf"  # <-- adapte si besoin
reader = PdfReader(pdf_path)

def clean_text(t: str) -> str:
    # Nettoyage léger : espaces, sauts de lignes multiples, hyphens de césure, etc.
    t = re.sub(r"-\n", "", t)                 # casse mots coupés par césure
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

# -------- 2) Extraction page par page + repérage de sections --------
pages = []
for i, page in enumerate(reader.pages, start=1):
    raw = page.extract_text() or ""
    pages.append({"page": i, "text": clean_text(raw)})

# Heuristique simple pour détecter des en-têtes/sections (Préambule, Article X, etc.)
SECTION_PAT = re.compile(r"^(Préambule|Article\s+\d+[^:\n]*|ANNEXE\s*\d+)", re.IGNORECASE)

def annotate_sections(pages):
    current_section = "Document"
    annotated = []
    for p in pages:
        # Cherche un titre de section au début de la page
        first_lines = p["text"].splitlines()[:12]
        section_found = None
        for line in first_lines:
            m = SECTION_PAT.match(line.strip())
            if m:
                section_found = m.group(0).strip()
                break
        if section_found:
            current_section = section_found
        annotated.append({**p, "section": current_section})
    return annotated

pages = annotate_sections(pages)

# -------- 3) Chunking du texte pour RAG --------
# Chunk ~800–1000 caractères avec overlap 150 pour du FR/longs articles
CHUNK_SIZE = 500
OVERLAP = 150

def chunk_text(text, page, section, size=CHUNK_SIZE, overlap=OVERLAP):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        # Essaie de couper sur un séparateur "propre" si possible
        slice_ = text[start:end]
        if end < n:
            # recule jusqu'au dernier point/fin de ligne pour éviter de couper une phrase
            cut = max(slice_.rfind("\n"), slice_.rfind(". "))
            if cut != -1 and cut > size * 0.5:
                end = start + cut + 1
                slice_ = text[start:end]
        chunks.append({
            "page": page,
            "section": section,
            "chunk": slice_.strip()
        })
        start = max(end - overlap, end)  # gère le cas overlap > restant
    return chunks

records = []
for p in pages:
    records.extend(chunk_text(p["text"], p["page"], p["section"]))

df = pd.DataFrame.from_records(records)
print(f"📄 Pages: {len(pages)} | 🧩 Chunks: {len(df)}")
print(df.head(2))

# -------- 4) Modèle d’embedding --------
# Multilingue, bon pour du FR
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Embeddings
df["embedding"] = df["chunk"].apply(lambda x: model.encode(x).tolist())

# (facultatif) Quelques métadonnées utiles pour filtrer
df["source_file"] = Path(pdf_path).name

# -------- 5) LanceDB : création/écriture --------
db = lancedb.connect("lancedb_reglement")     # dossier local
table = db.create_table(
    "reglement_chunks",
    data=df.to_dict(orient="records"),
    mode="overwrite"  # remet à zéro si existe
)

print(f"✅ Base créée : {len(df)} chunks insérés")
print("Champs : chunk, page, section, source_file, embedding")

# -------- 6) Exemple de recherche --------
# Pose ta requête en langage naturel (FR ok).
query = "Quelles sont les règles sur l'utilisation des téléphones pendant les examens ?"
query_vec = model.encode(query)

results = (
    table.search(query_vec, vector_column_name="embedding")
         .limit(5)
         .to_pandas()
)

# Affichage condensé
cols_to_show = ["section", "page", "chunk"]
print("\n🔍 Résultats similaires :")
for _, row in results[cols_to_show].iterrows():
    print(f"\n— {row['section']} (p.{int(row['page'])})")
    print(row["chunk"][:600] + ("…" if len(row["chunk"]) > 600 else ""))