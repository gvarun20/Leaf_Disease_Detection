# -*- coding: utf-8 -*-
# MLP for Pima Indians Dataset Serialize to JSON and HDF5
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os
import cv2

json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")


label=["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___Healthy",
       "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
       "Corn_(maize)___Healthy","Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot",
       "Grape___Esca_(Black_Measles)","Grape___Healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
       "Potato___Early_blight","Potato___Healthy","Potato___Late_blight","Tomato___Bacterial_spot",
       "Tomato___Early_blight","Tomato___Healthy","Tomato___Late_blight","Tomato___Leaf_Mold",
       "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
       "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus"]

test_image = image.load_img('im_for_testing_purpose/t.mosaicvirus.JPG', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)
print(result)
#print(result)
fresult=np.max(result)
label2=label[result.argmax()]
print(label2)
'''
a=np.round(result[0][0])
b=np.round(result[0][1])
c=np.round(result[0][2])
d=np.round(result[0][3])

print(a)
print(b)
print(c)
print(d)
'''

'''
label=["cats","dogs","horse","rose"]
test_image=cv2.imread("18.jpg")

tf=test_image.reshape(-1,test_image.shape[0],test_image.shape[1],test_image.shape[2])

prediction=loaded_model.predict_classes(tf)
fresult=np.max(prediction)
label2=label[prediction.argmax()]
print(label2)'''

'''
if result[0][0] == 1:
    prediction = 'dog'
    print(prediction)
elif result[0][1] == 1:
    prediction = 'cat'
    print(prediction)
elif result[0][2]== 1:
    prediction = 'Horse'
    print(prediction)
elif result[0][3] == 1:
    prediction = 'rose'
    print(prediction)
'''


