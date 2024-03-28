# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import joblib

# app = Flask(__name__)
# CORS(app)

# model_name = "gradient_boosting_ff_h1n1_vaccine_model.joblib"
# model=joblib.load(model_name)

# @app.route('/api/process_input', methods=["POST"])
# def process_input():
#     data = request.json
#     input_value = data.get('input')

#     for i in range(len(input_value)):
#         input_value[i] = int(input_value[i])
    
#     result = model.predict([input_value])

#     return jsonify({'message': f"Prediction: {input_value}"})

# if __name__ == '__main__':
#     app.run(debug=True, port=3001)

from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd  # Import pandas

app = Flask(__name__)
CORS(app)

model_name = "gradient_boosting_ff_h1n1_vaccine_model.joblib"
model = joblib.load(model_name)

@app.route('/api/process_input', methods=["POST"])
def process_input():
    data = request.json
    input_value = data.get('input')

    features_h1n1 = ['doctor_recc_h1n1', 'opinion_h1n1_risk', 'opinion_h1n1_vacc_effective', 'opinion_seas_risk', 'doctor_recc_seasonal',
                     'opinion_seas_vacc_effective', 'health_worker', 'h1n1_concern', 'health_insurance', 'h1n1_knowledge']
    
    input_df = pd.DataFrame([input_value], columns=features_h1n1)
    
    result = model.predict(input_df)

    return jsonify({'prediction': result.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=3001)