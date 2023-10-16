from flask import Flask
from flask import request, jsonify
import pickle

with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)
    f.close()

with open('model1.bin', 'rb') as f:
    model = pickle.load(f)
    f.close()

app = Flask('Homework-05')

@app.route('/predict', methods=['POST'])
def predict():
    sample = request.get_json()
    
    X = dv.transform([sample])
    y_pred = model.predict_proba(X)[0, 1]
    
    result = {
        'probability': float(y_pred),
    }
    
    return jsonify(result)
	
if __name__ == '__main__':
	 app.run(debug=True, host='0.0.0.0', port=9696)

