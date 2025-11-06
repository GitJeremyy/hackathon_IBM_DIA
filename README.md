# 🎓 School Assistant Chatbot - Hackathon IBM DIA

## 📋 Description

Assistant conversationnel intelligent pour les écoles **ESILV** (École Supérieure d'Ingénieurs Léonard de Vinci) et **EMLV** (École de Management Léonard de Vinci). Ce chatbot utilise l'intelligence artificielle et le traitement du langage naturel pour répondre aux questions des étudiants de manière contextuelle et précise.

Le projet combine :
- 🤖 **IBM Watsonx AI** (Llama-3-3-70B) pour la génération de réponses
- 🔍 **RAG (Retrieval Augmented Generation)** avec LanceDB pour la recherche vectorielle
- 🌐 **Interface Streamlit** pour une expérience utilisateur intuitive
- 🌍 **Support multilingue** (Français/Anglais)

---

## 🚀 Fonctionnalités

✅ **Sélection d'école** - Choisissez entre ESILV et EMLV  
✅ **Questions-réponses contextuelles** - Recherche vectorielle dans une base de connaissances  
✅ **Détection automatique de langue** - Répond dans la langue de la question  
✅ **Génération IA** - Utilise Llama-3 via IBM Watsonx pour des réponses naturelles  
✅ **Historique de conversation** - Suivi complet de l'échange  
✅ **Système de feedback** - Évaluation par étoiles et commentaires  
✅ **Interface moderne** - Design responsive avec CSS personnalisé  

---

## 🏗️ Architecture

```
hackathon_IBM_DIA/
├── app.py                          # Application Streamlit principale
├── source/
│   ├── load_QA.py                 # Chargement des Q&A dans LanceDB
│   ├── search_question.py         # Recherche vectorielle
│   ├── assistant.py               # Logique du chatbot avec IBM Watsonx
│   └── test.py                    # Tests
├── data/
│   └── Questions-Export-*.csv     # Base de données Q&A
├── lancedb_questions/             # Base de données vectorielle LanceDB
│   └── qa_table.lance/
├── prompts/
│   └── rag_prompt.txt            # Template de prompt RAG
├── certification/                 # Certificats et credentials
└── README.md
```

---

## 🔧 Installation

### Prérequis

- Python 3.11 ou 3.12
- pip
- Git

### Étapes d'installation

1. **Cloner le repository**
```bash
git clone https://github.com/GitJeremyy/hackathon_IBM_DIA.git
cd hackathon_IBM_DIA
```

2. **Créer un environnement virtuel**
```bash
# Windows (PowerShell)
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3.12 -m venv venv
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install streamlit pandas lancedb sentence-transformers ibm-watsonx-ai langdetect
```

4. **Configuration IBM Watsonx**

Éditez `source/assistant.py` et ajoutez vos credentials :
```python
API_KEY = "votre-api-key"
PROJECT_ID = "votre-project-id"
REGION = "eu-de"  # ou votre région
```

---

## 📊 Préparation des données

### 1. Charger la base de connaissances

Le fichier CSV doit contenir les colonnes suivantes :
- `Title` - Question
- `Content` - Réponse
- `Écoles` - École(s) concernée(s) (esilv, emlv)
- `Langues` - Langue(s) (Français, English)

```bash
python source/load_QA.py
```

Ce script :
1. Charge le CSV avec encodage ISO-8859-1
2. Génère des embeddings avec `multilingual-e5-base`
3. Stocke les vecteurs dans LanceDB
4. Crée les index pour la recherche vectorielle

### 2. Tester la recherche

```bash
python source/search_question.py
```

---

## 🎮 Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`

### Workflow utilisateur

1. **Sélection de l'école** - Choisir ESILV ou EMLV
2. **Conversation** - Poser des questions en français ou anglais
3. **Réponses IA** - Le chatbot répond en utilisant la base de connaissances
4. **Fermeture** - Clôturer la conversation
5. **Feedback** - Évaluer l'expérience (1-5 étoiles + commentaire)

### Exemple de questions

**ESILV (Français)**
- "Combien d'absences sont autorisées ?"
- "Comment fonctionne le système de notation ?"
- "Quels sont les horaires de la bibliothèque ?"

**EMLV (English)**
- "How many absences are allowed?"
- "What is the grading system?"
- "When is the library open?"

---

## 🧠 Fonctionnement technique

### Pipeline RAG (Retrieval Augmented Generation)

```
Question utilisateur
    ↓
Détection de langue (langdetect)
    ↓
Embedding de la question (multilingual-e5-base)
    ↓
Recherche vectorielle dans LanceDB (top 3 résultats)
    ↓
Filtrage par école et langue
    ↓
Construction du contexte
    ↓
Génération de réponse (Llama-3 via IBM Watsonx)
    ↓
Réponse finale à l'utilisateur
```

### Modèles utilisés

- **Embeddings** : `intfloat/multilingual-e5-base` (768 dimensions)
- **LLM** : `meta-llama/llama-3-3-70b-instruct` (IBM Watsonx)
- **Détection de langue** : `langdetect`

### Base de données vectorielle

- **LanceDB** - Base de données vectorielle open-source
- **Colonnes** :
  - `question` + `question_embedding` (768D)
  - `answer` + `answer_embedding` (768D)
  - `ecole` (esilv, emlv)
  - `langue` (Français, English)

---

## 🔑 Configuration

### Variables d'environnement (optionnel)

Créez un fichier `.env` :
```env
IBM_API_KEY=votre-api-key
IBM_PROJECT_ID=votre-project-id
IBM_REGION=eu-de
```

### Paramètres du modèle

Dans `assistant.py`, vous pouvez ajuster :
```python
params = {
    "max_new_tokens": 200,      # Longueur de la réponse
    "temperature": 0.6,         # Créativité (0-1)
    "repetition_penalty": 1.1   # Éviter les répétitions
}
```

---

## 📝 Structure des données

### Format CSV

```csv
Title;Content;Écoles;Langues
"Combien d'absences sont autorisées?";"Vous avez droit à 3 absences justifiées par semestre.";esilv,emlv;Français
"How many absences are allowed?";"You are allowed 3 justified absences per semester.";esilv,emlv;English
```

---

## 🐛 Dépannage

### Problème : Module non trouvé

```bash
# Vérifier que le venv est activé
pip list

# Réinstaller les dépendances
pip install streamlit pandas lancedb sentence-transformers ibm-watsonx-ai langdetect
```

### Problème : Erreur d'encodage CSV

Le script utilise `ISO-8859-1` par défaut. Si problème :
```python
df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
```

### Problème : Credentials IBM Watsonx

Vérifiez :
- API Key valide
- Project ID correct
- Région correcte (eu-de, us-south, etc.)

### Problème : LanceDB vide

Relancez le chargement :
```bash
python source/load_QA.py
```

---

## 🛠️ Développement

### Tests

```bash
python source/test.py
```

### Ajouter de nouvelles questions

1. Modifiez le CSV dans `data/`
2. Relancez `load_QA.py`
3. La base vectorielle sera mise à jour

### Personnaliser le prompt

Éditez `prompts/rag_prompt.txt` pour modifier le comportement du chatbot.

---

## 📈 Améliorations futures

- [ ] Authentification utilisateur
- [ ] Base de données PostgreSQL pour les feedbacks
- [ ] Support de fichiers PDF/DOCX
- [ ] Cache des réponses fréquentes
- [ ] Analytics et dashboards
- [ ] API REST
- [ ] Déploiement Docker
- [ ] CI/CD avec GitHub Actions

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Processus :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

---

## 📄 Licence

Ce projet a été développé dans le cadre du **Hackathon IBM DIA**.

---

## 👥 Équipe

Projet développé par l'équipe du Hackathon IBM DIA - Groupe A5

---

## 🔗 Liens utiles

- [IBM Watsonx Documentation](https://www.ibm.com/products/watsonx-ai)
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)

---

## 📞 Support

Pour toute question ou problème :
- Ouvrir une issue sur GitHub
- Contacter l'équipe via : **kryptosphere@devinci.fr**

---

**Made with ❤️ for ESILV & EMLV students**
