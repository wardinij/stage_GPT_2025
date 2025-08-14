import streamlit as st
import base64
import os

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

set_background("interface_utilisateur/wallpaper_3.jpg")  # remplace par ton fichier image

st.title("Gaindé Talent Provider (GTP)")

# Créer deux colonnes
col1, col2 = st.columns([2, 1])  # [largeur texte, largeur image]

with col1:
    st.markdown("""
    ### 
    Gaindé Talent Provider est une plateforme innovante qui facilite la recherche de codes HS et l’accès à des informations douanières précises.
    ### Notre mission
    - Simplifier le commerce international.
    - Offrir des outils rapides et fiables.
    - Aider les professionnels à gagner du temps.

    ### Contact
    📧 Email : contact@gaindetalent.com  
    📍 Adresse : 12, Rue Fila, Fann Hock, BP 6856 Dakar, Sénégal  
    🌐 Site web : [www.gaindetalentprovider.com](https://gaindetalent.com/)
    """)

with col2:
    photo_path = "interface_utilisateur/team.jpg"  # ton image
    if os.path.exists(photo_path):
        st.image(photo_path, use_container_width=True)
    else:
        st.error(f"Image introuvable : {photo_path}")
