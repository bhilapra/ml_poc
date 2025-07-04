from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        age = float(request.form["age"])
        rating = float(request.form["rating"])
        distance = float(request.form["distance"])
        order_type = request.form["order_type"]
        vehicle_type = request.form["vehicle_type"]

        # Build features exactly in this order:
        features = [
            age,
            rating,
            1 if order_type == "Drinks" else 0,
            1 if order_type == "Meal" else 0,
            1 if order_type == "Snack" else 0,
            1 if vehicle_type == "electric_scooter" else 0,
            1 if vehicle_type == "motorcycle" else 0,
            1 if vehicle_type == "scooter" else 0,
            distance
        ]

        X = np.array([features])
        X_scaled = scaler.transform(X)
        prediction = round(model.predict(X_scaled)[0], 2)

    return render_template("form.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
