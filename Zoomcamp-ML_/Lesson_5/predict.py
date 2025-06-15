from flask import Flask, request, jsonify
import pickle
import sklearn 
# import numpy as np



dv_file = '/Users/mac/Desktop/projects/zoomcamp ml/Lesson 5/dv.bin'
model_file = '/Users/mac/Desktop/projects/zoomcamp ml/Lesson 5/model1.bin'



with open(dv_file, 'rb') as f:
    dv = pickle.load(f)

with open(model_file, 'rb') as f:
    model = pickle.load(f)


app = Flask('predict')

@app.route('/predict', methods=['POST'])

def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    churn = y_pred >= 0.5

    result = {
        "churn_probability": float(y_pred),
        "churn": bool(churn)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)




