from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your model
model = load_model('mnist_model.h5')

# Route to the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to predict the digit
@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    img = Image.open(io.BytesIO(img.read())).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0  # Normalize
    
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    return jsonify({'digit': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)
