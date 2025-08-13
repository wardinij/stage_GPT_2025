import pandas as pd
import numpy as np
import faiss
import pickle
import subprocess
import time
from sentence_transformers import SentenceTransformer

# ========== CONFIGURATION ==========
ORIGINAL_CSV = "données/original.csv"  # Colonnes: "code", "description"
SAMPLE_SIZE = 50
TOP_K = 3
OLLAMA_MODEL = "gemma3"
OUTPUT_CSV = "données/reformulated_from_ollama.csv"

# ==================== TA FONCTION DE REFORMULATION ====================
def get_product_description_clean(description):
    prompt = f"""
    Reformulate the following product description in a professional, neutral, and concise style. 
    Keep the meaning unchanged. Only return the reformulated description — no explanation, no variants, no questions.

    Description:
    « {description} »
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
        response = ""
        for line in cleaned:
            if line.strip() and not line.lower().startswith(">>>"):
                response += line.strip() + " "

        return response.strip()
    except Exception as e:
        print("❌ Exception:", e)
        return None


# ==================== CONFIG MODELS ====================
MODELS = {
    "all-MiniLM": {
        "model": SentenceTransformer("all-MiniLM-L6-v2"),
        "index_path": "index_et_meta_data/index_faiss.idx",
        "meta_path": "index_et_meta_data/meta_data.pkl",
    },
    "BERT-base": {
        "model": SentenceTransformer("bert-base-uncased"),
        "index_path": "index_et_meta_data/index_faiss_bert.idx",
        "meta_path": "index_et_meta_data/meta_data_bert.pkl",
    },
    "all-mpnet-base-v2": {
        "model": SentenceTransformer("all-mpnet-base-v2"),
        "index_path": "index_et_meta_data/index_faiss_all_mpnet_base_v2.idx",
        "meta_path": "index_et_meta_data/meta_data_all_mpnet_base_v2.pkl",
    }
}

# ==================== CHARGEMENT DES DONNÉES ====================
df_original = pd.read_csv(ORIGINAL_CSV, dtype={"code": str})
df_original["code"] = df_original["code"].str.zfill(8)
sampled_df = df_original.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

# ==================== REFORMULATION VIA OLLAMA ====================
reformulated_texts = []
for i, row in sampled_df.iterrows():
    print(f"🔄 Reformulation {i+1}/{SAMPLE_SIZE} : {row['description']}")
    ref = get_product_description_clean(row["description"])
    if ref:
        print(f"   ➤ {ref}")
    else:
        print("   ⚠️  Reformulation échouée. Utilisation de l'originale.")
        ref = row["description"]
    reformulated_texts.append(ref)
    time.sleep(1)  # pour ne pas saturer Ollama

sampled_df["Reformulated_Description"] = reformulated_texts
sampled_df.rename(columns={"code": "HS_Code"}, inplace=True)
sampled_df.to_csv(OUTPUT_CSV, index=False)

# ==================== ÉVALUATION ====================
queries = sampled_df["Reformulated_Description"].tolist()
expected_codes = sampled_df["HS_Code"].tolist()

for model_name, config in MODELS.items():
    print(f"\n🔍 Évaluation du modèle : {model_name}")
    model = config["model"]
    index = faiss.read_index(config["index_path"])
    with open(config["meta_path"], "rb") as f:
        meta = pickle.load(f)
    codes = [str(c).zfill(8) for c in meta["codes"]]

    embeddings = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(embeddings, k=TOP_K)

    correct = 0
    for i in range(len(queries)):
        expected = expected_codes[i]
        predicted = [codes[idx] for idx in I[i]]
        if expected in predicted:
            correct += 1

    accuracy = correct / len(queries)
    print(f"✅ Précision top-{TOP_K} : {accuracy:.4f}")
