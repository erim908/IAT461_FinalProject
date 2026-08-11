import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Vibe Finder", layout="wide")

st.title("Vibe Finder: Audio-Based Playlist Explorer")
st.write(
    "Which songs sound alike, based only on their audio features and not "
    "their genre label? Adjust k in the sidebar to re-cluster live."
)

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]


VIBE_NAMES_K6 = {
    "0": "Instrumental / Electronic",
    "1": "Live-Sounding",
    "2": "Loud and Fast",
    "3": "Mellow / Acoustic",
    "4": "Feel-Good, Danceable",
    "5": "Speech / Rap-Heavy",
}


@st.cache_data
def load_data(path="spotify_songs.csv"):
    df = pd.read_csv(path)
    # same cleaning as the notebook: drop rows missing basic metadata,
    # dedupe on track_id, cap the loudness anomaly at 0 dB
    df = df.dropna(subset=["track_name", "track_artist", "track_album_name"]).copy()
    df = df.drop_duplicates(subset="track_id", keep="first").reset_index(drop=True)
    df["loudness"] = df["loudness"].clip(upper=0.0)
    return df


df = load_data()

with st.expander("Look at the cleaned data"):
    st.write(f"{len(df):,} songs after cleaning (deduped, missing metadata dropped, loudness capped)")
    st.dataframe(df[["track_name", "track_artist", "playlist_genre"] + AUDIO_FEATURES].head(20))

# ---- live clustering, same approach as the notebook ----
st.sidebar.header("Clustering")
k = st.sidebar.slider("Number of clusters (k)", min_value=5, max_value=10, value=6)

X = df[AUDIO_FEATURES].to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df["cluster"] = pd.Categorical([str(l) for l in labels])

sil = silhouette_score(X_scaled, labels)

col1, col2 = st.columns(2)
col1.metric("Songs analyzed", f"{len(df):,}")
col2.metric("Silhouette score", f"{sil:.3f}")

pca = PCA(n_components=2, random_state=42)
embedding = pca.fit_transform(X_scaled)
df["pc1"] = embedding[:, 0]
df["pc2"] = embedding[:, 1]

pca_tab, profile_tab, genre_tab, test_tab = st.tabs(
    ["PCA scatter", "Cluster profiles", "Genre mix", "Test a song"]
)

# PCA scatter of songs colored by cluster 
with pca_tab:
    st.caption(f"PC1 + PC2 explain {pca.explained_variance_ratio_.sum()*100:.1f}% of the variance in the 9 audio features.")
    fig_pca = px.scatter(
        df, x="pc1", y="pc2", color="cluster", opacity=0.4, height=550,
    )
    st.plotly_chart(fig_pca, width="stretch")

# cluster feature profiles 
with profile_tab:
    cluster_profile = df.groupby("cluster")[AUDIO_FEATURES].mean()
    overall_mean = df[AUDIO_FEATURES].mean()
    relative = (cluster_profile - overall_mean) / overall_mean.abs()

    if k == 6:
        st.caption("Cluster names below match the Checkpoint 2 notebook (only meaningful at k=6).")
        for c in sorted(cluster_profile.index, key=int):
            st.write(f"**Cluster {c}: {VIBE_NAMES_K6[c]}** ({(df['cluster'] == c).sum():,} songs)")
    else:
        st.caption("Vibe names were only worked out for k=6 (the notebook's chosen k). At other k values, just the profile numbers are shown below.")

    st.dataframe(cluster_profile.round(2))
    fig_profile = px.imshow(
        relative.T, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
        labels=dict(x="Cluster", color="vs. overall mean"), height=450,
    )
    st.plotly_chart(fig_profile, width="stretch")


with genre_tab:
    genre_mix = pd.crosstab(df["cluster"], df["playlist_genre"], normalize="index") * 100
    st.caption("Genre was never used to build the clusters, only checked afterward, same as in the notebook.")
    st.dataframe(genre_mix.round(1))
    fig_genre = px.imshow(
        genre_mix, text_auto=".0f", aspect="auto", color_continuous_scale="Blues", height=450,
    )
    st.plotly_chart(fig_genre, width="stretch")

# test a made-up song against the live model 
with test_tab:
    st.write("Set some audio features and see which cluster a song like this would land in.")
    c1, c2, c3 = st.columns(3)
    with c1:
        danceability = st.slider("danceability", 0.0, 1.0, 0.65)
        energy = st.slider("energy", 0.0, 1.0, 0.70)
        loudness = st.slider("loudness (dB)", -30.0, 0.0, -6.0)
    with c2:
        speechiness = st.slider("speechiness", 0.0, 1.0, 0.08)
        acousticness = st.slider("acousticness", 0.0, 1.0, 0.15)
        instrumentalness = st.slider("instrumentalness", 0.0, 1.0, 0.02)
    with c3:
        liveness = st.slider("liveness", 0.0, 1.0, 0.15)
        valence = st.slider("valence", 0.0, 1.0, 0.50)
        tempo = st.slider("tempo (BPM)", 50.0, 220.0, 120.0)

    if st.button("Find the closest cluster"):
        song = np.array([[danceability, energy, loudness, speechiness,
                           acousticness, instrumentalness, liveness, valence, tempo]])
        song_scaled = scaler.transform(song)
        pred = str(kmeans.predict(song_scaled)[0])
        if k == 6:
            st.success(f"Closest cluster: {pred} ({VIBE_NAMES_K6[pred]})")
        else:
            st.success(f"Closest cluster: {pred}")
        st.caption("This just finds the nearest of the k cluster centers. Cluster boundaries are soft, so treat it as a rough guide.")
