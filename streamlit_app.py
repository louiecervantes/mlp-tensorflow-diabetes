#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation



# Define the Streamlit app
def app():
    
    st.title('MLP Neural Network Using Tensorflow and Keras on the Diabetes dataset')
    st.subheader('by Louie F. Cervantes M.Eng., WVSU College of ICT')
    st.subheader('The Diabetes Dataset')

    st.write('The aim of this guide is to build a classification model to detect diabetes. \
        We will be using the diabetes dataset which contains 768 observations and \
        9 variables, as described below: \
        \npregnancies - Number of times pregnant. \
        \nglucose - Plasma glucose concentration. \
        \ndiastolic - Diastolic blood pressure (mm Hg). \
        \ntriceps - Skinfold thickness (mm). \
        \ninsulin - Hour serum insulin (mu U/ml). \
        \bmi – Basal metabolic rate (weight in kg/height in m). \
        \ndpf - Diabetes pedigree function. \
        \nage - Age in years. \
        \ndiabetes - "1"represents the presence of diabetes while "0" represents \
        the absence of it. This is the target variable.')
    
    #load the dataset
    df = pd.read_csv('diabetes.csv', header=0)
    st.dataframe(df, use_container_width=True)  
    #shufle the data
    df = df.sample(frac = 1)
    st.subheader('Configuring the Neural Net')
    with st.echo(code_location='below'):
        #set the number of hidden layers
        hidden_layers = st.slider('No. of hidden layers', 5, 15, 10)
        #set the number or iterations
        epochs = st.slider('Number of epochs', 50, 250, 100, 10)
        if st.button('Run the Classifier'):
            #load the data and the labels
            X = df.values[:,0:-1]
            y = df.values[:,-1].astype(int)
            
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Normalize the data using StandardScaler
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            # Define the model
            model = tf.keras.models.Sequential([
              tf.keras.layers.Dense(hidden_layers, activation='relu', input_shape=(X_train.shape[1],)),
              tf.keras.layers.Dense(1)
            ])

            # Compile the model
            model.compile(loss='mse', optimizer='adam', metrics=['mae'])

            # define a callback to write the output of each epoch to streamlit
            class EpochCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs={}):
                    st.text(f"Epoch {epoch+1}: loss={logs['loss']:.4f}, mae={logs['mae']:.4f}\n")

            callback = EpochCallback()
                        
            
            # Train the model
            history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, callbacks=[callback])  
           

            # Evaluate the model
            test_loss, test_mae = model.evaluate(X_test, y_test)
            st.write(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

            st.write('Making predictions on the test data...')
            # Make predictions on test data
            y_pred = model.predict(X_test)
            st.write('The coefficent of determination R^2 is computed.')
            # Calculate the coefficient of determination (R^2)
            r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
            st.write(f'R^2: {r2}')

            
    st.write('In this version of the MLP we used the Keras library running on Tensorflow.  \
            Keras is a high-level neural network library written in Python that can run \
            on top of TensorFlow, Theano, and other machine learning frameworks. \
            It was developed to make deep learning more accessible and easy to use \
            for researchers and developers.  TensorFlow provides a platform for \
            building and deploying machine learning models. It is designed to \
            be scalable and can be used to build models ranging from small experiments\
            to large-scale production systems. TensorFlow supports a wide range of \
            machine learning algorithms, including deep learning, linear regression, \
            logistic regression, decision trees, and many others.')        
   
#run the app
if __name__ == "__main__":
    app()
