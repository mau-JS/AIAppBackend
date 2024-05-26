from flask import Flask, request, jsonify
import pickle
import pandas as pd
from werkzeug.utils import secure_filename
import numpy as np
import os

# Load your trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from Post request
    data = request.get_json()
    # Make sure the data is in a 2D array if it's a single sample
    features = np.array([data['features']])
    # Make prediction
    prediction = model.predict(features)
    # Return results
    return jsonify({'prediction': prediction.tolist()})



#@app.route('/predict_csv', methods=['POST'])
#def predict_csv():
#    # Check if a file was posted
#    if 'file' not in request.files:
#        return jsonify({'error': 'No file part in the request'}), 400
#    file = request.files['file']
#    if file.filename == '':
#        return jsonify({'error': 'No file selected for uploading'}), 400
#    if file:
#        filename = secure_filename(file.filename)
#        
#        filepath = os.path.join('C:\\Users\\icono\\OneDrive\\Desktop', filename)  # change 'YourUsername' to your actual username
#        file.save(filepath)
#        # Load the CSV data
#        data = pd.read_csv(filepath)
#       # Make prediction
#       prediction = model.predict(data).tolist()
#        # Return results
#        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
