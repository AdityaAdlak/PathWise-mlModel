from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app , origins=["http://localhost:3000"])



model = joblib.load("model/career_model.pkl")
le_role = joblib.load("model/label_encoder_role.pkl")
le_stream = joblib.load("model/label_encoder_stream.pkl")
le_education = joblib.load("model/label_encoder_education.pkl")
le_time = joblib.load("model/label_encoder_time.pkl")
mlb_interests = joblib.load("model/mlb_interests.pkl")
mlb_skills = joblib.load("model/mlb_skills.pkl")

print("Model loaded successfully...")

allowed_skills = mlb_skills.classes_.tolist()
allowed_interests = mlb_interests.classes_.tolist()

@app.route("/options", methods=["GET"])
def get_allowed_options():
    return jsonify({
        "skills": mlb_skills.classes_.tolist(),
        "interests": mlb_interests.classes_.tolist()
    })
# so that give this same skills to the user


@app.route("/predict", methods=["POST"])
def predict_role():
    data = request.get_json()

    try:
        
        interests = mlb_interests.transform([data["interests"]])
        skills = mlb_skills.transform([data["skills"]])

        stream = le_stream.transform([data["stream"]])[0]
        education = le_education.transform([data["education"]])[0]
        time = le_time.transform([data["time"]])[0]

        # Combining all the features for inputs
        input_data = np.hstack([
            interests,
            skills,
            [[stream, education, time]]
        ])

        prediction = model.predict(input_data)[0]
        role = le_role.inverse_transform([prediction])[0]

        return jsonify({"predicted_role": role})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=5000)
