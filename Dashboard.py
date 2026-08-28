# dashboard_gironde_2025.py (version allégée et corrigée)
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import io
import gzip
from datetime import datetime

st.set_page_config(page_title="Dashboard Immobilier Gironde 2025", page_icon="🏘️", layout="wide")

# --- Dictionnaire des communes ---
COMMUNES_GIRONDE = {
    "33063": "Bordeaux", "33069": "Bruges", "33075": "Cenon", "33119": "Eysines",
    "33192": "Gradignan", "33200": "Gujan-Mestras", "33249": "Lormont", "33273": "Mérignac",
    "33281": "Pessac", "33312": "Saint-Médard-en-Jalles", "33318": "Talence", "33434": "Le Bouscat",
    "33449": "Villenave-d'Ornon", "33039": "Bègles", "33056": "Blanquefort", "33162": "Floirac",
    "33243": "Libourne", "33522": "Arcachon", "33529": "La Teste-de-Buch", "33550": "Cestas",
}

@st.cache_data(ttl=3600)
def load_gironde_2025_data():
    """
    Télécharge et charge uniquement les colonnes nécessaires depuis data.gouv.fr
    """
    url = "https://files.data.gouv.fr/geo-dvf/latest/csv/2025/departements/33.csv.gz"
    
    try:
        with st.spinner("📥 Téléchargement des données 2025..."):
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
        
        with st.spinner("🔄 Décompression et lecture (colonnes essentielles)..."):
            # Lire le CSV en mémoire
            with gzip.open(io.BytesIO(response.content), 'rt', encoding='utf-8') as f:
                # Lire l'en-tête pour identifier les colonnes disponibles
                first_line = f.readline()
                header = first_line.strip().split(',')
                f.seek(0)  # revenir au début
                
                # Colonnes utiles
                needed = ['date_mutation', 'valeur_fonciere', 'surface_reelle_bati',
                          'type_local', 'code_commune', 'code_postal', 
                          'latitude', 'longitude', 'nombre_pieces_principales']
                use_cols = [col for col in needed if col in header]
                if not use_cols:
                    st.error("Aucune colonne requise trouvée.")
                    return pd.DataFrame()
                
                # Recharger avec usecols
                df = pd.read_csv(f, sep=',', usecols=use_cols, low_memory=False)
        
        if df.empty:
            st.warning("Fichier vide")
            return pd.DataFrame()
        
        mem = round(df.memory_usage(deep=True).sum() / 1024**2, 1)
        st.sidebar.success(f"✅ {len(df):,} transactions chargées ({mem} Mo)")
        return df
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            st.error("🚫 Données 2025 pas encore disponibles")
        else:
            st.error(f"Erreur HTTP : {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur : {e}")
        return pd.DataFrame()

def prepare_data(df):
    if df.empty:
        return df
    df_clean = df.copy()
    
    # Dates
    if 'date_mutation' in df_clean.columns:
        df_clean["date_mutation"] = pd.to_datetime(df_clean["date_mutation"], errors='coerce')
    
    # Numériques
    if 'valeur_fonciere' in df_clean.columns:
        df_clean["valeur_fonciere"] = pd.to_numeric(df_clean["valeur_fonciere"], errors='coerce')
    if 'surface_reelle_bati' in df_clean.columns:
        df_clean["surface_reelle_bati"] = pd.to_numeric(df_clean["surface_reelle_bati"], errors='coerce')
    
    # Type de bien
    if 'type_local' in df_clean.columns:
        df_clean = df_clean[df_clean["type_local"].isin(['Maison', 'Appartement'])]
    
    # Suppression NA
    df_clean = df_clean.dropna(subset=['valeur_fonciere', 'surface_reelle_bati'])
    
    # Filtres de cohérence
    df_clean = df_clean[(df_clean['valeur_fonciere'] > 20000) & (df_clean['valeur_fonciere'] < 3000000)]
    df_clean = df_clean[(df_clean['surface_reelle_bati'] > 9) & (df_clean['surface_reelle_bati'] < 400)]
    
    # Prix m²
    df_clean['prix_m2'] = df_clean['valeur_fonciere'] / df_clean['surface_reelle_bati']
    df_clean = df_clean[(df_clean['prix_m2'] > 500) & (df_clean['prix_m2'] < 12000)]
    
    # Nom de commune
    if 'code_commune' in df_clean.columns:
        df_clean['code_commune'] = df_clean['code_commune'].astype(str).str.zfill(5)
        df_clean['nom_commune'] = df_clean['code_commune'].map(COMMUNES_GIRONDE)
        df_clean = df_clean.dropna(subset=['nom_commune'])
    
    return df_clean

# --- Interface ---
st.title("🏘️ Dashboard Immobilier Gironde - 2025 (allégé)")
st.markdown("Source : data.gouv.fr / DVF – Colonnes essentielles uniquement")

df_brut = load_gironde_2025_data()
if df_brut.empty:
    st.info("💡 Les données 2025 ne sont pas encore disponibles ou le téléchargement a échoué.")
    if st.button("🔄 Réessayer"):
        st.rerun()
    st.stop()

df = prepare_data(df_brut)
if df.empty:
    st.warning("Aucune transaction valide après nettoyage.")
    st.stop()

# --- Sélection commune ---
communes = sorted(df['nom_commune'].unique())
selected = st.sidebar.selectbox(
    "Choisissez une commune", 
    communes, 
    index=communes.index("Bordeaux") if "Bordeaux" in communes else 0
)
df_commune = df[df['nom_commune'] == selected].copy()
if df_commune.empty:
    st.stop()

# --- Filtres ---
st.sidebar.header("🔧 Filtres")
if 'code_postal' in df_commune.columns:
    cp_options = sorted(df_commune['code_postal'].astype(str).unique())
    cp_selection = st.sidebar.multiselect("Code postal", cp_options, default=cp_options)
else:
    cp_selection = []

type_local = st.sidebar.selectbox("Type de bien", ['Tous', 'Maison', 'Appartement'])
prix_min = st.sidebar.number_input("Prix min (€)", 0, step=20000)
prix_max = st.sidebar.number_input("Prix max (€)", int(df_commune['valeur_fonciere'].max()), step=50000)
surface_min = st.sidebar.slider("Surface min (m²)", 0, int(df_commune['surface_reelle_bati'].max()), 0)

df_filtre = df_commune.copy()
if cp_selection and 'code_postal' in df_filtre.columns:
    df_filtre = df_filtre[df_filtre['code_postal'].astype(str).isin(cp_selection)]
df_filtre = df_filtre[
    (df_filtre['valeur_fonciere'] >= prix_min) &
    (df_filtre['valeur_fonciere'] <= prix_max) &
    (df_filtre['surface_reelle_bati'] >= surface_min)
]
if type_local != 'Tous' and 'type_local' in df_filtre.columns:
    df_filtre = df_filtre[df_filtre['type_local'] == type_local]

if df_filtre.empty:
    st.warning("Aucune transaction ne correspond aux filtres.")
    st.stop()

# --- KPIs ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Prix moyen / m²", f"{df_filtre['prix_m2'].mean():,.0f} €")
col2.metric("Prix médian", f"{df_filtre['valeur_fonciere'].median():,.0f} €")
col3.metric("Transactions", f"{len(df_filtre):,}")
col4.metric("Surface moyenne", f"{df_filtre['surface_reelle_bati'].mean():.0f} m²")
if 'nombre_pieces_principales' in df_filtre.columns:
    col5.metric("Pièces principales", f"{df_filtre['nombre_pieces_principales'].mean():.1f}")

# --- Visualisations ---
c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(
        df_filtre, x='prix_m2', nbins=40,
        color='type_local' if 'type_local' in df_filtre.columns else None,
        marginal='box',
        title=f"Distribution des prix au m² - {selected}"
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.scatter(
        df_filtre, x='surface_reelle_bati', y='valeur_fonciere',
        color='type_local' if 'type_local' in df_filtre.columns else None,
        hover_data=['code_postal'],
        title="Surface vs Prix"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Carte (CORRIGÉE) ---
st.subheader(f"🗺️ Carte des transactions - {selected}")

if 'latitude' in df_filtre.columns and 'longitude' in df_filtre.columns:
    df_carte = df_filtre.copy()
    
    # Conversion robuste des coordonnées
    df_carte['latitude'] = pd.to_numeric(
        df_carte['latitude'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    )
    df_carte['longitude'] = pd.to_numeric(
        df_carte['longitude'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    )
    
    # Supprimer les NaN
    df_carte = df_carte.dropna(subset=['latitude', 'longitude'])
    
    if not df_carte.empty:
        if len(df_carte) > 500:
            df_carte = df_carte.sample(500)
            st.caption(f"Affichage de 500 transactions sur {len(df_filtre)} (échantillon aléatoire)")
        
        fig = px.scatter_mapbox(
            df_carte,
            lat='latitude',
            lon='longitude',
            color='prix_m2',
            size='surface_reelle_bati',
            hover_data={
                "valeur_fonciere": ":.0f",
                "type_local": True,
                "surface_reelle_bati": ":.0f",
                "prix_m2": ":.0f"
            },
            color_continuous_scale="RdYlGn_r",
            size_max=15,
            zoom=12,
            mapbox_style="open-street-map",
            title=f"Transactions à {selected} (2025)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📍 Aucune transaction avec coordonnées valides après nettoyage.")
else:
    st.info("📍 Colonnes latitude/longitude non disponibles.")

# --- Évolution temporelle ---
if 'date_mutation' in df_filtre.columns and not df_filtre.empty:
    df_filtre['mois'] = df_filtre['date_mutation'].dt.to_period('M')
    df_mensuel = df_filtre.groupby('mois').agg({
        'prix_m2': 'mean',
        'valeur_fonciere': ['count', 'mean']
    }).round(0)
    df_mensuel.columns = ['prix_m2_moyen', 'nb_transactions', 'prix_moyen']
    df_mensuel = df_mensuel.reset_index()
    df_mensuel['mois'] = df_mensuel['mois'].astype(str)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(df_mensuel, x='mois', y='prix_m2_moyen', markers=True, title="Évolution du prix au m²")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(df_mensuel, x='mois', y='nb_transactions', title="Nombre de transactions par mois")
        st.plotly_chart(fig, use_container_width=True)

# --- Top ventes ---
st.subheader("💰 Top 5 des ventes")
top = df_filtre.nlargest(5, 'valeur_fonciere')[['date_mutation', 'valeur_fonciere', 'surface_reelle_bati', 'prix_m2', 'type_local', 'code_postal']]
if not top.empty:
    top['valeur_fonciere'] = top['valeur_fonciere'].apply(lambda x: f"{x:,.0f} €")
    top['prix_m2'] = top['prix_m2'].apply(lambda x: f"{x:,.0f} €/m²")
    st.dataframe(top, hide_index=True, use_container_width=True)

st.subheader("📋 Dernières transactions")
display = df_filtre.sort_values('date_mutation', ascending=False).head(50)
display_cols = ['date_mutation', 'valeur_fonciere', 'surface_reelle_bati', 'prix_m2', 'type_local', 'code_postal']
available = [c for c in display_cols if c in display.columns]
for c in ['valeur_fonciere', 'prix_m2']:
    if c in display.columns:
        display[c] = display[c].apply(lambda x: f"{x:,.0f} €" + ("/m²" if c == 'prix_m2' else ""))
st.dataframe(display[available], hide_index=True, use_container_width=True)

st.markdown("---")
st.caption(f"Mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')} – Données DVF 2025 Gironde (33)")
