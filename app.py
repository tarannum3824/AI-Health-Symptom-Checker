


# from flask import Flask, render_template, request
# import pickle

# app = Flask(__name__)

# # load model
# model = pickle.load(open("disease_model.pkl","rb"))
# le = pickle.load(open("label_encoder.pkl","rb"))
# symptoms = pickle.load(open("symptoms_list.pkl","rb"))

# @app.route('/')
# def home():
#     return render_template("index.html", symptoms=symptoms)

# @app.route('/predict', methods=["POST"])
# def predict():

#     selected = request.form.getlist("symptoms")

#     vector = [0]*len(symptoms)

#     for s in selected:
#         if s in symptoms:
#             vector[symptoms.index(s)] = 1

#     prediction = model.predict([vector])
#     disease = le.inverse_transform(prediction)[0]

#     return render_template("index.html",
#                            prediction=disease,
#                            symptoms=symptoms)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# load model
model = pickle.load(open("disease_model.pkl","rb"))
le = pickle.load(open("label_encoder.pkl","rb"))
symptoms = pickle.load(open("symptoms_list.pkl","rb"))

# load precautions dataset
precaution_df = pd.read_csv("data/symptom_precaution.csv")

precaution_dict = {}

for _, row in precaution_df.iterrows():
    precaution_dict[row["Disease"]] = row[1:].dropna().tolist()


@app.route('/')
def home():
    return render_template("index.html", symptoms=symptoms)


@app.route('/predict', methods=["POST"])
def predict():

    selected = request.form.getlist("symptoms")

    vector = [0]*len(symptoms)

    for s in selected:
        if s in symptoms:
            vector[symptoms.index(s)] = 1

    prediction = model.predict([vector])
    disease = le.inverse_transform(prediction)[0]

    precautions = precaution_dict.get(disease, [])

    return render_template(
        "result.html",
        disease=disease,
        precautions=precautions
    )


if __name__ == "__main__":
    app.run(debug=True)