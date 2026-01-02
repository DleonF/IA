from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# =========================
# CARGAR DATASET
# =========================
df = pd.read_csv("spotify_dataset.csv")

# Features numéricas reales (NO strings)
features = [
    "Tempo",
    "Energy",
    "Danceability",
    "Positiveness",
    "Speechiness",
    "Liveness",
    "Acousticness",
    "Instrumentalness"
]

df_model = df[features + ["emotion", "Genre", "song", "Artist(s)"]].dropna()

X = df_model[features]
y = df_model["emotion"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_scaled, y)

# =========================
# MAPA DE GÉNEROS
# =========================
GENRE_MAP = {
    "Pop": ["Pop", "K-Pop", "J-Pop"],
    "Rock": ["Rock", "Alternative", "Classic Rock"],
    "Indie": ["Indie", "Indie Rock", "Indie Pop"],
    "Hip-Hop": ["Hip-Hop", "Rap", "Trap"]
}

# =========================
# RUTAS
# =========================

# Página 1 – Géneros principales
@app.route("/")
def generos():
    return render_template(
        "generos.html",
        generos=list(GENRE_MAP.keys())
    )

# Página 2 – Subgéneros
@app.route("/subgeneros/<genero>")
def subgeneros(genero):
    return render_template(
        "subgeneros.html",
        genero=genero,
        subgeneros=GENRE_MAP.get(genero, [])
    )

# Página 3 – Canciones + predicción
@app.route("/canciones/<genero>/<subgenero>", methods=["GET", "POST"])
def canciones(genero, subgenero):

    canciones = (
        df_model[
            df_model["Genre"]
            .str.contains(subgenero, case=False, na=False)
        ][["song", "Artist(s)"]]
        .drop_duplicates()
        .head(15)
        .values.tolist()
    )

    emocion = None
    cancion_sel = None

    if request.method == "POST":
        cancion_sel = request.form.get("song")

        fila = df_model[
            (df_model["Genre"].str.contains(subgenero, case=False, na=False)) &
            (df_model["song"] == cancion_sel)
        ].iloc[0]

        datos = fila[features].values.reshape(1, -1)
        datos_scaled = scaler.transform(datos)
        emocion = knn.predict(datos_scaled)[0]

    return render_template(
        "canciones.html",
        genero=genero,
        subgenero=subgenero,
        canciones=canciones,
        emocion=emocion,
        cancion_sel=cancion_sel
    )

if __name__ == "__main__":
    app.run(debug=True)
