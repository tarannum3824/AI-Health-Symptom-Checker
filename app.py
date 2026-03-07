


# import streamlit as st
# import pickle
# import pandas as pd

# # Load model files
# model = pickle.load(open("disease_model.pkl","rb"))
# le = pickle.load(open("label_encoder.pkl","rb"))
# symptoms = pickle.load(open("symptoms_list.pkl","rb"))

# # Load precautions
# precaution_df = pd.read_csv("../data/symptom_precaution.csv")

# precaution_dict = {}
# for _, row in precaution_df.iterrows():
#     precaution_dict[row["Disease"]] = row[1:].dropna().tolist()

# # Clean symptom names
# display_symptoms = [s.replace("_"," ").title() for s in symptoms]

# st.set_page_config(page_title="AI Health Checker", layout="wide")

# st.title("🩺 AI Health Symptom Checker")
# st.markdown("### Enter your symptoms to get possible disease prediction")

# # Symptom selector
# selected_display = st.multiselect(
#     "Select Symptoms",
#     display_symptoms
# )

# # convert back to original symptoms
# selected_symptoms = [
#     symptoms[display_symptoms.index(s)]
#     for s in selected_display
# ]

# # Convert to vector
# vector = [0]*len(symptoms)

# for s in selected_symptoms:
#     vector[symptoms.index(s)] = 1

# if st.button("Predict Disease"):

#     prediction = model.predict([vector])
#     disease = le.inverse_transform(prediction)[0]

#     st.success(f"Predicted Disease: {disease}")

#     # Show precautions
#     if disease in precaution_dict:

#         st.subheader("Recommended Precautions")

#         for p in precaution_dict[disease]:
#             st.write("•", p)


from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# load model
model = pickle.load(open("disease_model.pkl","rb"))
le = pickle.load(open("label_encoder.pkl","rb"))
symptoms = pickle.load(open("symptoms_list.pkl","rb"))

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

    return render_template("index.html",
                           prediction=disease,
                           symptoms=symptoms)

if __name__ == "__main__":
    app.run(debug=True)