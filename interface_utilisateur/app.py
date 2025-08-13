# app.py

import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import subprocess
import os
import pickle
import base64

# Juste après les imports, ajoute :
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = f.read()
    encoded = base64.b64encode(encoded).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("interface_utilisateur/background.jpg")  # remplace par ton fichier image

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

# Load FAISS index and metadata
index_path = "index_et_meta_data/index_faiss.idx"
meta_path = "index_et_meta_data/meta_data.pkl"

@st.cache_resource
def load_faiss_index():
    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return index, meta["codes"], meta["descriptions"]
    else:
        st.error("⚠️ Fichiers FAISS ou metadata manquants.")
        st.stop()

index, codes, descriptions = load_faiss_index()

# Description generator (Ollama)
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
            encoding='utf-8'
        )
        stdout, stderr = process.communicate(input=prompt, timeout=120)
        if process.returncode != 0:
            return None
        lines = [line.strip() for line in stdout.strip().split("\n") if line.strip()]
        response = " ".join(line for line in lines if not line.lower().startswith((">>>", "user:", "assistant:")))
        return response.strip()
    except subprocess.TimeoutExpired:
        return None

# FAISS search
def search_description(query, k=5):
    embedding = model.encode([query], convert_to_numpy=True)
    # Supprime cette normalisation si ton index FAISS ne l’utilise pas
    embedding = embedding / np.linalg.norm(embedding)
    distances, indices = index.search(embedding.reshape(1, -1), k)
    return [{"code": codes[i], "description": descriptions[i]} for i in indices[0]]

# Streamlit UI
st.title("🔍 Recherche de Code HS")
st.markdown("Entrez une **description** ou un **nom de produit** pour obtenir des codes HS similaires.")

mode = st.radio("Choisissez un mode :", ["Entrer une description", "Entrer un nom de produit"])

query = ""

if mode == "Entrer une description":
    query = st.text_input("✍️ Description générale du produit")
elif mode == "Entrer un nom de produit":
    product_name = st.text_input("📦 Nom du produit (ex : iPhone 15)")

if st.button("Rechercher des codes HS"):
    if mode == "Entrer un nom de produit" and product_name:
        with st.spinner("🧠 Génération de la description via Ollama..."):
            query = get_product_description_clean(product_name)
        if query:
            st.success(f"✅ Description générée : {query}")
        else:
            st.error("❌ Erreur lors de la génération de la description.")
            st.stop()
    elif mode == "Entrer une description" and not query:
        st.warning("⚠️ Veuillez entrer une description.")
        st.stop()

    with st.spinner("🔍 Recherche FAISS en cours..."):
        results = search_description(query)

    if results:
        st.markdown("### Résultats possibles")
        for r in results:
            st.markdown(f"**Code HS** : `{r['code']}`")
            st.markdown(f"**Description** : {r['description']}")
            st.markdown("---")
    else:
        st.warning("Aucun résultat trouvé.")
