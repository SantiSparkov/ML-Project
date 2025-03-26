from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Inicializar la aplicación FastAPI
app = FastAPI()

# Cargar el modelo previamente entrenado
model = load("best_random_forest_model.joblib")

# Listas de categorías para codificación
genre_list = [
    "Action", "Adventure", "Arcade", "Brawler", "Card & Board Game", "Fighting", "Indie",
    "MOBA", "Misc", "Music", "Pinball", "Platform", "Point-and-Click", "Puzzle", "RPG",
    "Racing", "Real Time Strategy", "Role-Playing", "Simulation", "Simulator", "Sport",
    "Sports", "Shooter", "Strategy", "Tactical", "Turn Based Strategy", "Visual Novel"
]

platform_list = [
    "2600", "NES", "PC", "X360", "3DS", "DS", "PS3", "PS2", "Wii", "PS4", "PS", "XOne",
    "GB", "GC", "N64", "SNES", "GEN", "DC", "XB", "GBA", "WiiU", "PSP", "PSV", "SAT",
    "WS", "3DO", "NG", "SCD", "PCFX", "TG16"
]

# Inicializar los LabelEncoders
genre_encoder = LabelEncoder()
genre_encoder.fit(genre_list)

platform_encoder = LabelEncoder()
platform_encoder.fit(platform_list)

# Modelo de entrada para predicción
class PredictionInput(BaseModel):
    Year: float
    Global: float
    Platform: str  # Platform como string
    Genre: str  # Genre como string
    Game_Age: float
    NorthAmerica_Global_Ratio: float
    Europe_Global_Ratio: float
    Japan_Global_Ratio: float
    RestOfWorld_Global_Ratio: float

# Endpoint raíz
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de predicción con Random Forest"}

@app.post("/predict/")
def predict(input_data: PredictionInput):
    try:
        # Convertir `Genre` de texto a su representación numérica
        try:
            genre_encoded = genre_encoder.transform([input_data.Genre])[0]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"El género '{input_data.Genre}' no es válido. Géneros permitidos: {genre_list}"
            )

        # Convertir `Platform` de texto a su representación numérica
        try:
            platform_encoded = platform_encoder.transform([input_data.Platform])[0]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"La plataforma '{input_data.Platform}' no es válida. Plataformas permitidas: {platform_list}"
            )

        # Crear un DataFrame con los datos de entrada
        input_df = pd.DataFrame([{
            "Year": input_data.Year,
            "Global": input_data.Global,
            "Platform": platform_encoded,
            "Genre": genre_encoded,
            "Game_Age": input_data.Game_Age,
            "NorthAmerica_Global_Ratio": input_data.NorthAmerica_Global_Ratio,
            "Europe_Global_Ratio": input_data.Europe_Global_Ratio,
            "Japan_Global_Ratio": input_data.Japan_Global_Ratio,
            "RestOfWorld_Global_Ratio": input_data.RestOfWorld_Global_Ratio
        }])

        # Validar dimensiones de entrada
        if input_df.shape[1] != model.n_features_in_:
            raise HTTPException(
                status_code=400,
                detail=f"El modelo espera {model.n_features_in_} características, pero se proporcionaron {input_df.shape[1]}"
            )

        # Asegurar que los datos son `float64`
        input_df = input_df.astype("float64")

        # Hacer predicción
        prediction = model.predict(input_df)

        # Devolver el resultado
        return {"prediction": prediction.tolist()}

    except Exception as e:
        print(f"Error durante la predicción: {e}")
        raise HTTPException(status_code=500, detail="Error durante la predicción. Verifica los datos de entrada.")