from flask import Flask, request, jsonify
import numpy as np
import pickle
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar modelo
with open("modelo.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "mensaje": "Bienvenido a la API de prediccion de diabetes",
        "endpoints": {
            "/predict": "Realiza una prediccion con 10 parametros (edad, sexo, imc, presion, s1-s6)",
            "/retrain": "Reentrena el modelo con los datos originales",
            "/info": "Muestra informacion tecnica del modelo"
        },
        "parametros_para_predict": [
            "edad", "sexo", "imc", "presion",
            "s1", "s2", "s3", "s4", "s5", "s6"
        ],
        "notas": {
            "sexo": {
                "tipo": "float",
                "ejemplos": {
                    "hombre": 0.050680,
                    "mujer": -0.044642
                },
                "nota": "Este valor está normalizado; no se debe usar texto como 'male' o 'female'"
            }
        },
        "ejemplo": "/predict?edad=0.03&sexo=0.01&imc=0.05&presion=0.02&s1=-0.01&s2=0.03&s3=-0.02&s4=0.01&s5=0.04&s6=0.01"
    })

@app.route("/predict", methods=["GET"])
def predict():
    if not request.args:
        return jsonify({
            "instrucciones": "Este endpoint predice la progresión de la diabetes.",
            "como_usarlo": "Debes enviar 10 parámetros por la URL:",
            "parametros": {
                "edad": "float",
                "sexo": "float",
                "imc": "float",
                "presion": "float",
                "s1": "float",
                "s2": "float",
                "s3": "float",
                "s4": "float",
                "s5": "float",
                "s6": "float"
            },
            "ejemplo": "/predict?edad=0.03&sexo=0.050680&imc=0.05&presion=0.02&s1=-0.01&s2=0.03&s3=-0.02&s4=0.01&s5=0.04&s6=-0.01"
        })

    try:
        params = [
            "edad", "sexo", "imc", "presion", 
            "s1", "s2", "s3", "s4", "s5", "s6"
        ]

        features = []
        for param in params:
            value = request.args.get(param)
            if value is None:
                return jsonify({"error": f"Falta el parámetro: {param}"}), 400
            features.append(float(value))

        features_array = np.array([features])
        pred = model.predict(features_array)[0]
        pred = round(float(pred), 2)

        # Interpretación del resultado
        if pred < 100:
            interpretacion = "Bajo riesgo de progresion"
        elif 100 <= pred <= 170:
            interpretacion = "Riesgo medio de progresion"
        else:
            interpretacion = "Alto riesgo de progresion"

        return jsonify({
            "prediccion_diabetes": pred,
            "interpretacion": interpretacion
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "modelo": "Random Forest Regressor",
        "descripcion": "Predice la progresion de la diabetes a partir de 10 caracteristicas clinicas.",
        "parametros_requeridos_para_predict": [
            "edad", "sexo", "imc", "presion", "s1", "s2", "s3", "s4", "s5", "s6"
        ],
        "formato_predict": "/predict?edad=...&sexo=...&imc=...&presion=...&s1=...&s2=...&s3=...&s4=...&s5=...&s6=...",
        "formato_retrain": "/retrain (GET sin parámetros)"
    })

@app.route("/retrain", methods=["GET"])
def retrain():
    try:
        # Cargar el dataset
        diabetes = load_diabetes()
        X = diabetes.data
        y = diabetes.target

        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar nuevo modelo
        new_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        new_model.fit(X_train, y_train)

        # Guardar modelo
        with open("modelo.pkl", "wb") as f:
            pickle.dump(new_model, f)

        # Recargar el modelo global
        global model
        model = new_model

        # Hacer predicciones y calcular métricas
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return jsonify({
            "message": "Modelo reentrenado exitosamente",
            "metricas": {
                "mean_squared_error": round(mse, 2),
                "r2_score": round(r2, 4)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
