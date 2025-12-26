from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

encoder = LabelEncoder()

@app.route("/", methods=["GET", "POST"])
def index():
    recommendation = ""

    if request.method == "POST":
        skin_type = request.form["skin_type"]
        skin_tone = request.form["skin_tone"]
        concern = request.form["concern"]

        df = pd.read_csv("data.csv")

        encoder.fit(df["skin_type"])
        skin_type = encoder.transform([skin_type])[0]

        encoder.fit(df["skin_tone"])
        skin_tone = encoder.transform([skin_tone])[0]

        encoder.fit(df["concern"])
        concern = encoder.transform([concern])[0]

        prediction = model.predict([[skin_type, skin_tone, concern]])
        recommendation = prediction[0]

    return render_template("index.html", recommendation=recommendation)

if __name__ == "__main__":
    app.run(debug=True)
