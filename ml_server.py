# creating a basic flask server
import json
import tensorflow as tf
import numpy as np

from flask import Flask, request

app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')

# feature model will take the input same as 'model' but give the output of all layers
model = tf.keras.models.load_model('model.h5')
feature_model = tf.keras.models.Model(
    model.inputs,
    [layer.output for layer in model.layers]
)

_ , (x_test, _) = tf.keras.datasets.mnist.load_data()
x_test = x_test/ 255

def get_prediction():
    index = np.random.choice(x_test.shape[0])
    image = x_test[index, :,:]
    image_arr = np.reshape(image,(1,x_test.shape[1]*x_test.shape[2]))
    return feature_model.predict((image_arr)), image


@app.route('/', methods = ['GET','POST'])
def index():
    if request.method == 'POST':
        preds, image = get_prediction()
        final_preds = [p.tolist() for p in preds]
        return json.dumps({
            'prediction': final_preds,
            'image': image.tolist()
        })
    return "Welcome to ML server"

if __name__ == '__main__':
    app.run()

# web: sh setup.sh && streamlit run ml_server.py && streamlit run app.py
