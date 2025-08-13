import pandas as pd
import torch
from transformers import AutoTokenizer, T5EncoderModel
import numpy as np
import faiss
from tqdm import tqdm
import subprocess
import os
import pickle

# === Config device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", device)

# === Charger modèle T5 (encodeur uniquement) ===
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5EncoderModel.from_pretrained(model_name).to(device)
model.eval()

# === Fonction d'encodage ===
def encode_bert(sentences, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Encodage"):
        batch = sentences[i:i+batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        ).to(device)

        with torch.no_grad():
            output = model(**encoded).last_hidden_state  # [batch, seq, hidden]
            mask = encoded['attention_mask'].unsqueeze(-1).expand(output.size())
            masked_output = output * mask
            summed = masked_output.sum(dim=1)
            counts = mask.sum(dim=1)
            mean_pooled = summed / counts

            embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(embeddings)

# === 3. Encoder ou charger index ===
index_path = "index_et_meta_data/index_faiss_t5.idx"
meta_path = "index_et_meta_data/meta_data_t5.pkl"

if os.path.exists(index_path) and os.path.exists(meta_path):
    print("📂 Chargement de l'index FAISS et des métadonnées...")
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
        codes = meta["codes"]
        descriptions = meta["descriptions"]
else:
    print("⚙️  Encodage des descriptions et création de l'index FAISS...")
    df = pd.read_csv(
        "données/original.csv",
        encoding="utf-8"
    )
    descriptions = df["description"].astype(str).tolist()
    codes = df["code"].astype(str).tolist()

    embeddings = encode_bert(descriptions)

    # Normalisation pour similarité cosinus
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"codes": codes, "descriptions": descriptions}, f)

    print("✅ Index FAISS et données sauvegardés.")

# === Fonction description produit (Ollama) ===
def get_product_description_clean(product_name):
    prompt = f"""
Give a short, general, and formal description of the following product: « {product_name} ».
Start with the product category name, followed by a sentence describing its use in a clear and neutral way.
Do not include the model name or technical specifications.
"""
    try:
        process = subprocess.Popen(
            ['ollama', 'run', 'gemma3'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        stdout, stderr = process.communicate(input=prompt, timeout=120)

        if process.returncode != 0:
            print("❌ Error running Ollama:", stderr)
            return None

        cleaned = stdout.strip().split("\n")
        response = " ".join(
            line.strip()
            for line in cleaned
            if line.strip() and not line.lower().startswith(">>>")
        )

        return response.strip()

    except subprocess.TimeoutExpired:
        print("❌ Ollama took too long to respond.")
        process.kill()
        return None

# === Recherche dans FAISS ===
def search_description(query, k=5):
    embedding = encode_bert([query])
    faiss.normalize_L2(embedding)
    distances, indices = index.search(embedding, k)

    return [
        {"code": codes[idx], "description": descriptions[idx]}
        for idx in indices[0]
    ]

# === Boucle interactive ===
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
