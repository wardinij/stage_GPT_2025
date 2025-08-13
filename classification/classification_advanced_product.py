import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm
import subprocess

# === 1. Charger le CSV ===
df = pd.read_csv("données/original.csv") # Assure-toi qu'il y a bien 'description' et 'code'
descriptions = df["description"].tolist()
codes = df["code"].tolist()
print(df.head())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", device)

# === 2. Charger le modèle SentenceTransformer ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === 3. Encoder les descriptions existantes ===
embeddings = model.encode(descriptions, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity
index.add(embeddings)

# === 4. Fonction pour transformer un nom de produit en description avec Ollama ===

def get_product_description_clean(product_name):
    prompt = f"""
Give a short, general, and formal description of the following product: « {product_name} ».
Start with the product category name, followed by a comma, then a sentence describing its use in a clear and neutral way.
Do not include the model name or technical specifications.
"""

    try:
        process = subprocess.Popen(
            ['ollama', 'run', 'gemma3'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'  # ← ← ← ajoute ceci pour corriger l'erreur
        )

        stdout, stderr = process.communicate(input=prompt, timeout=120)

        if process.returncode != 0:
            print("❌ Error running Ollama:", stderr)
            return None

        cleaned = stdout.strip().split("\n")
        response = ""
        for line in cleaned:
            if line.strip() and not line.lower().startswith(">>>"):
                response += line.strip() + " "

        return response.strip()

    except subprocess.TimeoutExpired:
        print("❌ Ollama took too long to respond.")
        process.kill()
        return None

# === 5. Fonction de recherche FAISS ===
def search_description(query, k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    distances, indices = index.search(query_embedding.reshape(1, -1), k)

    results = []
    for idx in indices[0]:
        results.append({
            "code": codes[idx],
            "description": descriptions[idx]
        })
    return results


while True:
    mode = input("\nEnter mode — (1) General description, (2) Product name, or 'quit' to exit: ").strip().lower()
    
    if mode in ["quit", "exit"]:
        print("Goodbye! 👋")
        break

    if mode == "1":
        query = input("Enter a general product description: ").strip()
    elif mode == "2":
        product_name = input("Enter product name (e.g. iPhone 15): ").strip()
        print("Generating general description with Ollama...")
        query = get_product_description_clean(product_name)
        if query:
            print(f"✔️ Generated description: {query}")
        else:
            print("⚠️ No response generated.")
            continue
    else:
        print("Invalid choice. Please type 1, 2, or 'quit'.")
        continue

    print("🔎 Searching for HS code...")
    results = search_description(query, k=5)

    print("\n=== Search Results ===")
    for r in results:
        print(f"Code: {r['code']}\nDescription: {r['description']}\n")
