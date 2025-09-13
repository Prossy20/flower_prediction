
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.form.to_dict()
    features = [float(data['sepal_length']), float(data['sepal_width']), float(data['petal_length']), float(data['petal_width'])]
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Map prediction to Iris species
    iris_species = ['Setosa', 'Versicolour', 'Virginica']
    result = iris_species[prediction[0]]
    
    return render_template('index.html', prediction_text=f'Predicted Species: {result}')

if __name__ == "__main__":
    app.run(debug=True)
