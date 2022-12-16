import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
import math

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    feature = [np.array(float_features)]
    prediction = model.predict(feature)
    #output = round(prediction[0],2)
    return render_template("index.html", prediction_text = "${}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
