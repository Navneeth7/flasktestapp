from flask import Flask
import pickle
import numpy as np

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
@app.route("/prediction",methods=["POST"])
def hello():
    data=request.get_json()
    print(data)
    prediction=ml(data['n'],data['p'],data['k'],data['temp'],data['hum'],data['ph'])
    return prediction[0]
if __name__ == '__main__':
    app.debug = True
    app.run()
