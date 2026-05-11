import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('data/data.csv')

# Features used for ML
features = ['danceability', 'energy', 'loudness',
            'speechiness', 'acousticness',
            'instrumentalness', 'liveness',
            'valence', 'tempo']

df = df.dropna(subset=features)

# Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

# ---------- Cosine Similarity ----------
similarity = cosine_similarity(scaled)

def recommend_cosine(song_name):
    if song_name not in df['name'].values:
        return ["Song not found"]

    idx = df[df['name'] == song_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    return [df.iloc[i[0]]['name'] for i in scores]


# ---------- KMeans Clustering ----------
kmeans = KMeans(n_clusters=20, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled)

def recommend_cluster(song_name):
    if song_name not in df['name'].values:
        return ["Song not found"]

    cluster_id = df[df['name'] == song_name]['cluster'].values[0]
    cluster_songs = df[df['cluster'] == cluster_id]['name'].values[:6]

    return cluster_songs