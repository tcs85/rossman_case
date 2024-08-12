import pickle
import os
import pandas as pd 

from flask import Flask, request, Response
from webapp.rossmann.rossmann import rossmann


# Load the trained model from a pickle file
model = pickle.load(open('model/model_rossmann.pkl', 'rb'))

# Initialize the Flask API
app = Flask(__name__)

# Define the API endpoint for predictions
@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    # Get the JSON data from the POST request
    test_json = request.get_json()
    
    # Check if there is data in the JSON
    if test_json:
        # If the data is a dictionary (single example), convert it to a DataFrame with one row
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        # If the data is a list of dictionaries (multiple examples), convert it to a DataFrame
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
    
    # Instantiate the Rossmann class
    pipeline = rossmann()
    
    # Data cleaning step
    df1 = pipeline.data_cleanning(test_raw)
    
    # Feature engineering step
    df2 = pipeline.feature_engineering(df1)
    
    # Data preparation step
    df3 = pipeline.data_preparation(df2)
    
    # Get predictions
    response = pipeline.get_prediction(model, test_raw, df3)
    
    # Return the predictions as a JSON response
    return Response(response, status=200, mimetype='application/json')

# Run the Flask app
if __name__ == '__main__':
    port=os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
    app.run('0.0.0.0')
