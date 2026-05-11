import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Music Recommender")

st.title("🎵 Music Recommendation System")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    cols = ['name','artists','danceability','energy','loudness',
            'speechiness','acousticness','instrumentalness',
            'liveness','valence','tempo']
    
    df = pd.read_csv('data/data.csv', usecols=cols)
    df = df.sample(3000, random_state=42)
    df = df.reset_index(drop=True)   # ⭐ IMPORTANT FIX
    return df

df = load_data()
st.success("✅ Dataset Loaded Successfully")

st.write("### Preview")
st.dataframe(df.head())

# ---------------- FEATURE SCALING ----------------
features = ['danceability','energy','loudness','speechiness',
            'acousticness','instrumentalness','liveness',
            'valence','tempo']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# ---------------- KMEANS CLUSTERING ----------------
kmeans = KMeans(n_clusters=8, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

score = silhouette_score(X_scaled, df['cluster'])
st.write(f"### 📏 Silhouette Score of Clustering: {score:.3f}")

# ---------------- RECOMMENDATION FUNCTIONS ----------------
def recommend(song_name, n=5):
    song_index = df[df['name'] == song_name].index[0]
    distances = cosine_similarity(X_scaled, X_scaled[song_index].reshape(1, -1))
    song_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
    recommended = []

    for i in song_list[1:n+1]:
        recommended.append(df.iloc[i[0]][['name','artists']])

    return pd.DataFrame(recommended)

def recommend_from_cluster(song_name, n=5):
    cluster_id = df[df['name'] == song_name]['cluster'].values[0]
    cluster_songs = df[df['cluster'] == cluster_id][['name','artists']]
    return cluster_songs.sample(n)

# ---------------- UI SECTION ----------------
st.write("## 🎧 Get Recommendations")

song_list = df['name'].values
selected_song = st.selectbox("Choose a song", song_list)

num = st.slider("Number of recommendations", 3, 10, 5)

col1, col2 = st.columns(2)

with col1:
    if st.button("Recommend by Cosine Similarity"):
        recs = recommend(selected_song, num)
        st.subheader("🎯 Similar Songs")
        st.table(recs)

with col2:
    if st.button("Recommend from Same Cluster"):
        recs = recommend_from_cluster(selected_song, num)
        st.subheader("🧩 Songs from Same Cluster")
        st.table(recs)