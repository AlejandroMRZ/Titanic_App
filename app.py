from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# 📦 Cargar el modelo entrenado
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🏠 Página de inicio con el formulario
@app.route('/')
def home():
    return render_template('form.html')

# 🔮 Predicción desde el formulario web
@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        # 1️⃣ Extraer datos del formulario
        pclass = int(request.form['Pclass'])
        sex = 1 if request.form['Sex'] == 'female' else 0
        age = float(request.form['Age'])

        # 2️⃣ Realizar la predicción
        input_data = [[pclass, sex, age]]
        prediction = model.predict(input_data)[0]
        resultado = "¡Sobrevivió! 🎉" if prediction == 1 else "No sobrevivió. 😢"

        # 3️⃣ Devolver el resultado en el formulario
        return render_template('form.html', prediction=resultado)

    except Exception as e:
        return render_template('form.html', prediction=f"Error: {str(e)}")

# 🔮 Predicción desde la API JSON (para Postman u otras herramientas)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # 1️⃣ Extraer datos del JSON
        pclass = int(data.get('Pclass'))
        sex = 1 if data.get('Sex').lower() == 'female' else 0
        age = float(data.get('Age'))

        # 2️⃣ Realizar la predicción
        input_data = [[pclass, sex, age]]
        prediction = model.predict(input_data)[0]

        # 3️⃣ Devolver la predicción en formato JSON
        return jsonify({
            "predicción": "Sobrevivió" if prediction == 1 else "No sobrevivió"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
