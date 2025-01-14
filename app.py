from flask import Flask, request, jsonify, render_template
import pickle
from dotenv import load_dotenv
import os
import google.generativeai as genai  # Correct SDK for Gemini

# 📂 Load environment variables
load_dotenv()

# 🔑 Load Google Gemini API Key
GEMINI_API_KEY = os.getenv("GOOGLE_AI_STUDIO")

# ⚠️ Check if the API Key is loaded
if not GEMINI_API_KEY:
    raise ValueError("❌ ERROR: Gemini API Key not found. Check your .env file.")
else:
    print("✅ Gemini API Key Loaded.")

# 🚀 Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# 🏷️ Initialize Flask app
app = Flask(__name__)

# 📦 Load trained Titanic model
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🧠 Initialize Gemini model (fast or pro)
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')  # Use 'gemini-pro' for more depth

# 🔮 Function to generate explanation using Gemini
def get_gemini_explanation(pclass, sex, age, prediction):
    prediction_text = "sobrevivió" if prediction == 1 else "no sobrevivió"
    sex_text = "femenino" if sex == 1 else "masculino"

    # 📖 Dynamic prompt
    prompt = (
        "Eres un historiador experto en el Titanic y tienes acceso a un modelo de Machine Learning que predice la supervivencia de los pasajeros. "
        "Utilizando las siguientes características, explica de forma lógica y detallada por qué este pasajero habría sobrevivido o no:\n\n"
        f"🛳️ **Datos del pasajero:**\n"
        f"- Clase: {pclass}\n"
        f"- Género: {sex_text}\n"
        f"- Edad: {age} años\n"
        f"- Predicción del modelo: {prediction_text}\n\n"
        "📖 **Instrucciones:**\n"
        "- Proporciona un análisis histórico y lógico basado en estos datos.\n"
        "- Relaciona la clase social y el acceso a los botes salvavidas.\n"
        "- Sé claro, conciso y profesional.\n\n"
        "📢 **Explicación:**"
    )

    try:
        # 💬 Generate explanation using Gemini
        response = gemini_model.generate_content(prompt)
        return response.text if response.text else "Sin explicación disponible."
    except Exception as e:
        return f"Error al obtener la explicación de la IA: {str(e)}"

# 🏠 Home route
@app.route('/')
def home():
    return render_template('form.html')

# 📊 Prediction route (Form Submission)
@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        # 📥 Extract user inputs
        pclass = int(request.form['Pclass'])
        sex = 1 if request.form['Sex'] == 'female' else 0
        age = float(request.form['Age'])

        # 🤖 Model Prediction
        input_data = [[pclass, sex, age]]
        prediction = model.predict(input_data)[0]
        result = "¡Sobrevivió! 🎉" if prediction == 1 else "No sobrevivió. 😢"

        # 🔮 Generate AI Explanation
        explanation = get_gemini_explanation(pclass, sex, age, prediction)

        return render_template('form.html', prediction=result, explanation=explanation)

    except Exception as e:
        return render_template('form.html', prediction=f"Error: {str(e)}")

# 🔌 API Prediction Route (For JSON Requests)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # 📥 Extract data from JSON
        pclass = int(data.get('Pclass'))
        sex = 1 if data.get('Sex').lower() == 'female' else 0
        age = float(data.get('Age'))

        # 🤖 Model Prediction
        input_data = [[pclass, sex, age]]
        prediction = model.predict(input_data)[0]

        # 🔮 Generate AI Explanation
        explanation = get_gemini_explanation(pclass, sex, age, prediction)

        return jsonify({
            "predicción": "Sobrevivió" if prediction == 1 else "No sobrevivió",
            "explicación": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# 🔥 Run the app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

