# %%
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import cv2
import base64
import requests
import os
import pickle
import numpy as np
from sklearn import metrics
from keras.models import load_model

url = "http://localhost:8000/api/animal/vector"
num_classes = 5  # Change this to the number of classes you have

def img2vec(img):
    v, buffer = cv2.imencode(".png", img)
    img_str = base64.b64encode(buffer)
    data = "image data,"+str.split(str(img_str),"'")[1]
    response = requests.get(url, json={"img":data})

    return response.json()["Hog"]
img_list =[]
#%%

path = "dataset\\train"
i = 0
for subfolder in os.listdir(path):
    for f in os.listdir(os.path.join(path,subfolder)):
        img = cv2.imread(os.path.join(path,subfolder)+"/"+f)
        img22 = []
        img22 = img2vec(img)
        img22.append(i)
        img_list.append(img22) 
    i += 1

#%%
write_path = "ImageFeatureTrain.pkl"
pickle.dump(img_list, open(write_path,"wb"))
print("data preparation is done")

#%%
animalVectors = pickle.load(open('ImageFeatureTrain.pkl','rb'))
animalVectors_np = np.array(animalVectors)

X_train = animalVectors_np[:,0:-1]
Y_train = animalVectors_np[:,-1]

# %%
print(X_train)

# %%
print(Y_train)

#%%
path = "Dataset\\test"
img_list =[]
i = 0

for subfolder in os.listdir(path):
    for f in os.listdir(os.path.join(path,subfolder)):
        img = cv2.imread(os.path.join(path,subfolder)+"/"+f)
        
        img22 = img2vec(img)
        img22.append(i)
        img_list.append(img22)
    i += 1


#%%
write_path = "ImageFeatureTest.pkl"
pickle.dump(img_list, open(write_path,"wb"))
print("data preparation is done")

#%%
animalVectors = pickle.load(open('ImageFeatureTest.pkl','rb'))
animalVectors_np = np.array(animalVectors)
X_test = animalVectors_np[:,0:-1]
Y_test = animalVectors_np[:,-1]

# %%
print(X_test)

# %%
print(Y_test)

# %%
print(X_train.shape[1])
print(Y_train.shape[1])
# %%
model = models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu' ))
model.add(layers.Dense(128, activation='relu' ))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%%
model.fit(X_train, Y_train, epochs=10)
# %%
print(model.predict(X_test))
# %%
model = load_model(r'../model/AnimalImageFeatureModel.h5')
Y_test = Y_test.astype(int)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
print(y_pred)

# %%
print("Accuracy:",metrics.accuracy_score(Y_test , y_pred))
print("Confusion Matrix:", metrics.confusion_matrix(Y_test , y_pred))

# %%
path_model = "AnimalImageFeatureModel.pk"
pickle.dump(model, open(path_model,"wb"))
model.save('AnimalImageFeatureModel.h5')
# %%
