from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
model = joblib.load("social_match_rf_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()
    features = [
        "same_selected_activity", "shared_activities_count", "intent_match",
        "time_overlap_selected_activity", "preferred_time_block_match", "same_city",
        "past_successful_sessions", "personality_match_score", "kindness_score_diff",
        "average_rating_of_other_user", "mutual_connections", "last_connected_days_ago",
        "is_online_now"
    ]
    X = [[input_data[f] for f in features]]
    probability = model.predict_proba(X)[0][1]
    return jsonify({ "score": round(probability, 4) })

@app.route("/", methods=["GET"])
def index():
    return "ML Model is running!"

# 🚨 THIS PART IS REQUIRED FOR CLOUD RUN
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
