# For deployment

import streamlit as st
import json
import requests
import matplotlib.pyplot as plt
import numpy as np

# use streamlit run app.py to run the app

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


def predict():
    preds, image = get_prediction()
    final_preds = [p.tolist() for p in preds]
    return {
        'prediction': final_preds,
        'image': image.tolist()
    }

def main():
    st.title('Neural Network Visualiser')
    st.sidebar.markdown("## Input Image")

    if st.button('Get random prediction'):
        response = predict()
        # response = requests.post(URI, data={})
        # response = json.loads(response.text)
        preds = response.get('prediction')
        image = response.get('image')
        image = np.reshape(image,(28,28))
        
        st.sidebar.image(image,width = 150)
        
        for layer, p in enumerate(preds):
            numbers = np.squeeze(np.array(p))
            plt.figure(figsize = (32,4))
            
            # output layer
            if layer == 2:
                row = 1
                col = 10
            
            # hidden layers
            else: 
                row = 2
                col = 16
            
            for i, number in enumerate(numbers):
                plt.subplot(row,col, i+1)
                plt.imshow(number * np.ones((8,8,3)).astype('float32'))
                plt.xticks([])
                plt.yticks([])
                
                if layer == 2:
                    plt.xlabel(str(i), fontsize  = 40)
            
            plt.subplots_adjust(wspace = 0.05, hspace = 0.05)
            plt.tight_layout()
            st.text('Layer {}'.format(layer+1))
            st.pyplot()

if __name__=='__main__': 
    main()