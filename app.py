from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)


df = pd.read_csv("spotify_dataset.csv") #Dataset

# Features numéricas reales
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

df_model = df[features + ["emotion", "Genre", "song", "Artist(s)"]].dropna() # elimina filas faltantes

X = df_model[features] #in
y = df_model["emotion"] #out

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=7) #Knn con 7 vecinos parecidas
knn.fit(X_scaled, y)

EMOTION_MAP = {
    "joy": "Alegria",
    "sadness": "Tristeza",
    "anger": "Enojo",
    "fear": "Miedo",
    "love": "Amor",
    "surprise": "Sorpresa",
    "neutral": "Neutral",
    "angry": "Enojo",
    "confusion": "Confusion",
    "interest": "Interes",
    "thirst": "Sed",
    "pink": "Rosa",
    "true": "Verdadero"
}
EMOTION_EXCLUDE = {"angry", "pink", "thirst"}
EMOTIONS = sorted(
    e for e in df_model["emotion"].dropna().unique().tolist()
    if e not in EMOTION_EXCLUDE
)

# Generos a subgeneros
GENRE_MAP = {
    "Pop": ["Pop", "K-Pop", "J-Pop"],
    "Rock": ["Rock", "Alternative", "Classic Rock"],
    "Indie": ["Indie", "Indie Rock", "Indie Pop"],
    "Hip-Hop": ["Hip-Hop", "Rap", "Trap"]
}
GENRE_LOOKUP = {
    sub.lower(): main
    for main, subs in GENRE_MAP.items()
    for sub in subs
}


# RUTAS ::  Pagina 1 –> Generos
@app.route("/", methods=["GET", "POST"])
def generos():
    q = request.values.get("q", "").strip()
    resultados = []
    emocion = None
    confianza = None
    cancion_sel = None
    artista_sel = None

    if q:
        matches = df_model[
            df_model["song"].str.contains(q, case=False, na=False) |
            df_model["Artist(s)"].str.contains(q, case=False, na=False)
        ][["song", "Artist(s)", "Genre"]].drop_duplicates()

        matches["main_genre"] = matches["Genre"].str.lower().map(GENRE_LOOKUP)
        matches = matches[matches["main_genre"].notna()]

        resultados = [
            {
                "song": row["song"],
                "artist": row["Artist(s)"],
                "genero": row["main_genre"],
                "subgenero": row["Genre"]
            }
            for _, row in matches.iterrows()
        ]

    if request.method == "POST":
        cancion_sel = request.form.get("song")
        artista_sel = request.form.get("artist")
        subgenero_sel = request.form.get("subgenero")

        fila = df_model[
            (df_model["Genre"].str.contains(subgenero_sel or "", case=False, na=False)) &
            (df_model["song"] == cancion_sel) &
            (df_model["Artist(s)"] == artista_sel)
        ].iloc[0]

        datos = fila[features].values.reshape(1, -1)
        datos_scaled = scaler.transform(datos)
        emocion_raw = knn.predict(datos_scaled)[0]
        
        proba = knn.predict_proba(datos_scaled)[0]  #vector de probabilidad con sus 7 vecinos
        confianza = float(proba.max())     #elige la mayor probabilidad
        emocion = EMOTION_MAP.get(emocion_raw, emocion_raw)

    return render_template(
        "generos.html",
        generos=list(GENRE_MAP.keys()),
        emociones=[
            {"value": e, "label": EMOTION_MAP.get(e, e)}
            for e in EMOTIONS
        ],
        q=q,
        resultados=resultados,
        emocion=emocion,
        confianza=confianza,
        cancion_sel=cancion_sel,
        artista_sel=artista_sel
    )

# Pagina 2 –> Subgeneros
@app.route("/subgeneros/<genero>")
def subgeneros(genero):
    return render_template(
        "subgeneros.html",
        genero=genero,
        subgeneros=GENRE_MAP.get(genero, [])
    )

# Pagina 3 –> Canciones
@app.route("/canciones/<genero>/<subgenero>", methods=["GET", "POST"])
def canciones(genero, subgenero):

    base = df_model[
        df_model["Genre"]
        .str.contains(subgenero, case=False, na=False)
    ]

    base = base[["song", "Artist(s)"]].drop_duplicates()

    if len(base) > 15:
        base = base.sample(n=15)

    canciones = base.values.tolist()

    emocion = None
    confianza = None
    cancion_sel = None

    if request.method == "POST":
        cancion_sel = request.form.get("song")

        fila = df_model[
            (df_model["Genre"].str.contains(subgenero, case=False, na=False)) &
            (df_model["song"] == cancion_sel)
        ].iloc[0]

        datos = fila[features].values.reshape(1, -1)
        datos_scaled = scaler.transform(datos)
        emocion_raw = knn.predict(datos_scaled)[0]
        proba = knn.predict_proba(datos_scaled)[0]
        confianza = float(proba.max())
        emocion = EMOTION_MAP.get(emocion_raw, emocion_raw)

    return render_template(
        "canciones.html",
        genero=genero,
        subgenero=subgenero,
        canciones=canciones,
        emocion=emocion,
        confianza=confianza,
        cancion_sel=cancion_sel
    )

# Pagina -> Emociones
@app.route("/emociones/<emocion>")
def emociones(emocion):
    base = df_model[df_model["emotion"] == emocion]
    base = base[features + ["song", "Artist(s)"]].drop_duplicates()

    emotion_index = None
    for i, label in enumerate(knn.classes_):
        if label == emocion:
            emotion_index = i
            break

    canciones = []
    for _, row in base.iterrows():
        datos = row[features].values.reshape(1, -1)
        datos_scaled = scaler.transform(datos)
        proba = knn.predict_proba(datos_scaled)[0]
        confianza = float(proba[emotion_index]) if emotion_index is not None else None
        canciones.append([row["song"], row["Artist(s)"], confianza])
    canciones.sort(key=lambda item: item[2] if item[2] is not None else -1, reverse=True)
    canciones = canciones[:5]

    return render_template(
        "emociones.html",
        emocion=EMOTION_MAP.get(emocion, emocion),
        emocion_raw=emocion,
        canciones=canciones
    )

if __name__ == "__main__":
    app.run(debug=True)
