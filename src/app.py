from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

from utils import db_connect
engine = db_connect()

# your code here
app = Flask(__name__)


# Cargar modelo
with open('modelo.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Cargar scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.form

    try:
        # Extraer los valores del formulario
        valores = [float(datos[var]) for var in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'Age', 'DiabetesPedigreeFunction'
        ]]

        entrada = np.array([valores])
        entrada_esc = scaler.transform(entrada)
        prediccion = modelo.predict(entrada_esc)[0]

        return render_template('index.html', prediccion=prediccion)
    
    except Exception as e:
        return f"Error en la predicción: {e}"

if __name__ == '__main__':
    app.run(debug=True)