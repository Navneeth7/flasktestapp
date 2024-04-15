from flask import Flask
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify,request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*")
# @app.route('/api/data')
# def get_data():
#     data = {'message': 'Hello from the backend!'}
#     return jsonify(data)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

app = Flask(__name__)
def ml(n,p,k,temp,hum,ph):
    file_path = 'random_forest_model.pkl'
    with open(file_path , 'rb') as f:
        dict1 = pickle.load(f)
    data = np.array([[n, p, k,temp,hum,ph]])
    prediction = dict1.predict(data)
    return prediction

def ml2(temp,hum,mois,soilType,cropType,n,p,k):
    with open('preprocessor.pkl', 'rb') as preprocessor_file, open('model.pkl', 'rb') as model_file:
        preprocessor = pickle.load(preprocessor_file)
        model = pickle.load(model_file)
    new_sample = {
    'Temparature': temp,
    'Humidity': hum,
    'Moisture': mois,
    'Soil Type': soilType,
    'Crop Type': cropType,
    'Nitrogen': n,
    'Potassium': p,
    'Phosphorous': k
    }
    new_sample_df = pd.DataFrame([new_sample])

    # Transform the new sample with the loaded preprocessor
    new_sample_transformed = preprocessor.transform(new_sample_df)

    # Use the loaded model to predict
    prediction = model.predict(new_sample_transformed)

    print(f"Predicted Fertilizer Name: {prediction[0]}")
    return prediction[0]
@app.route("/prediction",methods=["POST"])
def hello():
    data=request.get_json()
    print(data)
    prediction=ml(data['n'],data['p'],data['k'],data['temp'],data['hum'],data['ph'])
    return prediction[0]
@app.route("/fertprediction",methods=["POST"])
def fert():
    data=request.get_json()
    print(data)
    prediction=ml2(data['temp'],data['hum'],data['mois'],data['soilType'],data['cropType'],data['n'],data['p'],data['k'])
    print(prediction)
    return prediction
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
