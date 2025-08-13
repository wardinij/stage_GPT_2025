import pandas as pd
import numpy as np
import faiss
import pickle
import subprocess
import time
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, T5EncoderModel

# ========== CONFIGURATION ==========
ORIGINAL_CSV = "données/original.csv"  # Colonnes: "code", "description"
SAMPLE_SIZE = 50
TOP_K = 3
OLLAMA_MODEL = "gemma3"
OUTPUT_CSV = "données/reformulated_from_ollama.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== FONCTION DE REFORMULATION ====================
def get_product_description_clean(description):
    prompt = f"""
Reformulate the following product description in a professional, neutral, and concise style. 
Keep the meaning unchanged. Only return the reformulated description — no explanation, no variants, no questions.

Description:
« {description} »
"""
    try:
        process = subprocess.Popen(
            ['ollama', 'run', OLLAMA_MODEL],
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

    except Exception as e:
        print("❌ Exception:", e)
        return None

# ==================== INITIALISATION DES MODELES ====================
# Pour BERT
bert_model_name = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name).to(device).eval()

def encode_bert(texts):
    embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        encoded = bert_tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
        with torch.no_grad():
            output = bert_model(**encoded).last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1).expand(output.size())
            summed = (output * mask).sum(dim=1)
            counts = mask.sum(dim=1)
            mean_pooled = summed / counts
        embeddings.append(mean_pooled.cpu().numpy())
    return np.vstack(embeddings)

# Pour Flan-T5
t5_model_name = "google/flan-t5-small"
t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
t5_model = T5EncoderModel.from_pretrained(t5_model_name).to(device).eval()

def encode_t5(texts):
    embeddings = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        encoded = t5_tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128).to(device)
        with torch.no_grad():
            output = t5_model(**encoded).last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1).expand(output.size())
            summed = (output * mask).sum(dim=1)
            counts = mask.sum(dim=1)
            mean_pooled = summed / counts
        embeddings.append(mean_pooled.cpu().numpy())
    return np.vstack(embeddings)

# SentenceTransformer models
model_all_minilm = SentenceTransformer("all-MiniLM-L6-v2")
model_mpnet = SentenceTransformer("all-mpnet-base-v2")

MODELS = {
    "all-MiniLM": {
        "encode": lambda texts: model_all_minilm.encode(texts, convert_to_numpy=True, normalize_embeddings=True),
        "index_path": "index_et_meta_data/index_faiss.idx",
        "meta_path": "index_et_meta_data/meta_data.pkl",
    },
    "BERT-base": {
        "encode": lambda texts: encode_bert(texts),
        "index_path": "index_et_meta_data/index_faiss_bert.idx",
        "meta_path": "index_et_meta_data/meta_data_bert.pkl",
    },
    "all-mpnet-base-v2": {
        "encode": lambda texts: model_mpnet.encode(texts, convert_to_numpy=True, normalize_embeddings=True),
        "index_path": "index_et_meta_data/index_faiss_all_mpnet_base_v2.idx",
        "meta_path": "index_et_meta_data/meta_data_all_mpnet_base_v2.pkl",
    },
    "google/flan-t5-small": {
        "encode": lambda texts: encode_t5(texts),
        "index_path": "index_et_meta_data/index_faiss_t5.idx",
        "meta_path": "index_et_meta_data/meta_data_t5.pkl",
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
    time.sleep(1)

sampled_df["Reformulated_Description"] = reformulated_texts
sampled_df.rename(columns={"code": "HS_Code"}, inplace=True)
sampled_df.to_csv(OUTPUT_CSV, index=False)

# ==================== ÉVALUATION ====================
queries = sampled_df["Reformulated_Description"].tolist()
expected_codes = sampled_df["HS_Code"].tolist()

for model_name, config in MODELS.items():
    print(f"\n🔍 Évaluation du modèle : {model_name}")
    encode_fn = config["encode"]

    index = faiss.read_index(config["index_path"])
    with open(config["meta_path"], "rb") as f:
        meta = pickle.load(f)
    codes = [str(c).zfill(8) for c in meta["codes"]]

    embeddings = encode_fn(queries)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    # Normalisation FAISS si nécessaire
    if "normalize_embeddings" not in config or not getattr(config, "normalize_embeddings", False):
        faiss.normalize_L2(embeddings)

    D, I = index.search(embeddings, k=TOP_K)

    correct = 0
    for i in range(len(queries)):
        expected = expected_codes[i]
        predicted = [codes[idx] for idx in I[i]]
        if expected in predicted:
            correct += 1

    accuracy = correct / len(queries)
    print(f"✅ Précision top-{TOP_K} : {accuracy:.4f}")
