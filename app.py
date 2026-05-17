import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Music Recommender")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1e1e2f, #2b5876);
    color: white;
}

/* Headings */
h1, h2, h3, h4, label {
    color: white !important;
}

/* Buttons */
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

/* Table fix */
[data-testid="stTable"] {
    background-color: white;
    color: black;
    border-radius: 10px;
    padding: 10px;
}

/* Dataframe fix */
[data-testid="stDataFrame"] {
    background-color: white;
    color: black;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

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


# ---------------- FEATURE SCALING ----------------
features = ['danceability','energy','loudness','speechiness',
            'acousticness','instrumentalness','liveness',
            'valence','tempo']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# ---------------- KMEANS CLUSTERING ----------------
kmeans = KMeans(n_clusters=8, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)


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

if st.button("🎯 Recommend Similar Songs"):
    recs1 = recommend(selected_song, num//2)
    recs2 = recommend_from_cluster(selected_song, num - num//2)

    final_recs = pd.concat([recs1, recs2]).drop_duplicates().head(num)

    st.subheader("🎵 Recommended Songs For You")
    st.table(final_recs)