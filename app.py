from flask import Flask, request, jsonify, render_template
import pickle
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Load the Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

# 📦 Load the trained Titanic model
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🤖 Function to get AI explanation from Google Gemini
def get_gemini_explanation(pclass, sex, age, prediction):
    prediction_text = "sobrevivió" if prediction == 1 else "no sobrevivió"
    sex_text = "femenino" if sex == 1 else "masculino"

    prompt = (
        f"Basado en los datos de un pasajero del Titanic:\n"
        f"- Clase: {pclass}\n"
        f"- Género: {sex_text}\n"
        f"- Edad: {age} años\n\n"
        f"Explícame por qué este pasajero {prediction_text} al hundimiento del Titanic."
    )

    api_url = "https://api.google.com/gemini/generate"  # Replace with the correct endpoint

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt,
        "max_tokens": 150
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json().get("text", "Sin explicación disponible.")
    else:
        return "Error al obtener la explicación de la IA."

# 🏠 Home route to render the form
@app.route('/')
def home():
    return render_template('form.html')

# 🔮 Prediction route with AI explanation
@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        # 1️⃣ Extract form data
        pclass = int(request.form['Pclass'])
        sex = 1 if request.form['Sex'] == 'female' else 0
        age = float(request.form['Age'])

        # 2️⃣ Make a prediction
        input_data = [[pclass, sex, age]]
        prediction = model.predict(input_data)[0]
        result = "¡Sobrevivió! 🎉" if prediction == 1 else "No sobrevivió. 😢"

        # 3️⃣ Get AI explanation
        explanation = get_gemini_explanation(pclass, sex, age, prediction)

        # 4️⃣ Render the result with explanation
        return render_template('form.html', prediction=result, explanation=explanation)

    except Exception as e:
        return render_template('form.html', prediction=f"Error: {str(e)}")

# 🔮 Prediction API route for JSON data
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        pclass = int(data.get('Pclass'))
        sex = 1 if data.get('Sex').lower() == 'female' else 0
        age = float(data.get('Age'))

        input_data = [[pclass, sex, age]]
        prediction = model.predict(input_data)[0]

        explanation = get_gemini_explanation(pclass, sex, age, prediction)

        return jsonify({
            "predicción": "Sobrevivió" if prediction == 1 else "No sobrevivió",
            "explicación": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
