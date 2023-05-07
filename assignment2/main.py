from fastapi import FastAPI, File
import keras
import numpy as np
from PIL import Image

# Load the pre-trained model
model = keras.models.load_model('cuisine_model.h5')

# Create a FastAPI application
app = FastAPI()

# Define a route that accepts POST requests with uploaded image files and returns the prediction
@app.post('/predict')
async def predict(file: bytes = File(...)):
    # Convert the file to a PIL Image object
    img = Image.open(io.BytesIO(file))
    # Resize the image to the size required by the model
    img = img.resize((224, 224))
    # Convert the image to a numpy array
    img_arr = np.array(img)
    # Convert the image from RGB format to BGR format (consistent with the pre-trained model)
    img_arr = img_arr[:, :, ::-1]
    # Scale the values of the image to be between 0 and 1 (consistent with the preprocessing of the pre-trained model)
    img_arr = img_arr.astype('float32') / 255.0
    # Add an extra dimension to the image to match the input shape of the model
    img_arr = np.expand_dims(img_arr, axis=0)
    # Use the pre-trained model for prediction
    pred = model.predict(img_arr)[0]
    # Calculate the probability of Other class
    other_prob = 1 - np.sum(pred)
    # Convert the prediction result to a dictionary and return it
    result = {'Japanese': pred[0], 'Italian': pred[1], 'Chinese': pred[2], 'French': pred[3], 'Other': other_prob}
    return result
