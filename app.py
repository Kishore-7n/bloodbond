from flask import Flask, request, jsonify
from flask_cors import CORS 
import joblib


# Load the trained model
model = joblib.load("health_tips_model.pkl")

# Initialize Flask app
app = Flask(__name__)

CORS(app) 

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()
    
    # Extract features
    blood_group = data['blood_group']
    hemoglobin_level = data['hemoglobin_level']
    disease = data['disease']
    
    # Combine features into a single string
    combined_features = f"{blood_group} {hemoglobin_level} {disease}"
    
    # Make a prediction
    prediction = model.predict([combined_features])
    
    # Return the prediction as a JSON response
    return jsonify({'healthy_tips': prediction[0]})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True , port=8000)