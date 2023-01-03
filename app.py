# Before creating this .py file, the prediction models were built in google colab, using pycaret library
# the related file created in google colab for building models is saved as total_comfort.ipynb
# First step in model building is to build it in google colab using pycaret library
# and pickle out .pkl file of the model and save it in the same director as this .py file
# as a next step given in this .py file, the .pkl model is loaded and used for predictions based on user input
# this file will be run in anaconda command prompt using command: streamlit run comfort_using_pkl.py
# the same file can then be deployed on internet web using streamlit sharing

# ---- HIDE STREAMLIT STYLE ----
#hide_st_style = """
#            <style>
#            #MainMenu {visibility: hidden;}
#            footer {visibility: hidden;}
#            header {visibility: hidden;}
#            </style>
#            """
#st.markdown(hide_st_style, unsafe_allow_html=True)


from pycaret.regression import load_model
from pycaret.regression import predict_model

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# to load the models from the same PC folder where this .py file is saved.
# It is assumed that the models with .pkl extension are already saved in the same folder
# the .pkl files comprising the models were created in google colab using pycaret library and then pickled out
# and saved
loaded_model = pickle.load(open('rf_fgsm_model3.pkl', 'rb'))

#etr stands for extra tree regressor

# Creating the Titles and Image
st.title("Predicting GSM of Knitted Fabric")

#making columns
col1, col2 = st.columns(2)

# taking user input
yarn_count = col1.slider('Yarn Count (Ne)', 16, 32, 24)
sl = col2.slider('Stitch Length', 0.21, 0.39, 0.25)
dial = col1.slider('Machine Dia', 16, 34, 20)
gauge = col2.slider('Machine Gauge', 20, 28, 25)
#contentID = st.slider('ContentID', 1, 5, 1)
#routeID = st.slider('RouteID', 1, 8, 1)

option1 = st.selectbox(
     'Select Fabric Content',
     ('100%Cotton', 
        '60%Cotton40%Polyester',
         '55%Poly34%Ctn11%Rayon',
         '9%Cotton10%Polyester',
         '50%Cotton50%Polyester')
     )

#st.write('You selected:', option1)

if option1 == '100%Cotton':
     contentID = 1
if option1 == '60%Cotton40%Polyester':
     contentID = 2
if option1 == '55%Poly34%Ctn11%Rayon':
     contentID = 3
if option1 == '9%Cotton10%Polyester':
     contentID = 4
if option1 == '50%Cotton50%Polyester':
     contentID = 5

#....
option2 = st.selectbox(
     'Select Process Route',
     ('Stretch,Dry,Compact(Tubular)', 
        'Slit,Stretch-Dry,Compact(Open)',
         'Stretch,Dry,Re-Stretch,Dry,Compact(Tubular)',
         'Stretch,Dry,Compact(Tubular),Slit,Stretch-Dry,Compact(Open)',
         'Stretch,Dry,Re-Stretch,Dry,Compact(Tubular),Slit,Stretch-Dry,Sp.Application,Compact(Open)',
         'Slit,Stretch-Dry,SpApplication,Compact(Open)',
         'Slit,Stretch-Dry(Relax Dry)Compact(Felt Belt),Stretch,Dry,Compact(Tubular)',
         'Slit,Stretch-Dry(Open)'
         )
     )


if option2 == 'Stretch,Dry,Compact(Tubular)':
     routeID = 1
if option2 == 'Slit,Stretch-Dry,Compact(Open)':
     routeID = 2
if option2 == 'Stretch,Dry,Re-Stretch,Dry,Compact(Tubular)':
     routeID = 3
if option2 == 'Stretch,Dry,Compact(Tubular),Slit,Stretch-Dry,Compact(Open)':
     routeID = 4
if option2 == 'Stretch,Dry,Re-Stretch,Dry,Compact(Tubular),Slit,Stretch-Dry,Sp.Application,Compact(Open)':
     routeID = 5
if option2 == 'Slit,Stretch-Dry,SpApplication,Compact(Open)':
     routeID = 6
if option2 == 'Slit,Stretch-Dry(Relax Dry)Compact(Felt Belt),Stretch,Dry,Compact(Tubular)':
     routeID = 7
if option2 == 'Slit,Stretch-Dry(Open)':
     routeID = 8

# store the inputs

features = [sl, yarn_count, dial, gauge, contentID, routeID]

# convert user inputs into an array for the model

# convert user inputs into an array for the model
#in float(x) argument, word float is used to recognize decimal/float values of SL
#if we use int(x) argument, integer values from user input will be taken, while float values will be taken as 0 (zero)
int_features = [float(x) for x in features]
final_features = [np.array(int_features)]


# Final prediction
prediction_rf = loaded_model.predict(final_features)

#st.success(f'Total Cloting Comfort: {round(prediction_etr[0], 2)}')

st.metric(label = "Predicted Finished Fabric GSM", value = int(prediction_rf))





