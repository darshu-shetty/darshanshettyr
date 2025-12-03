#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import keras
import cv2
import os
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize
# from keras.models import sequential
from keras.layers import Conv2D , MaxPooling2D
from keras.layers import Activation , Dropout , Flatten , Dense
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")


# Data preprocessing

# In[2]:


s=len(os.listdir('Downloads/colored_DR_1000_each/severe'))
print(s)


# In[3]:


import os
import cv2
import matplotlib.pyplot as plt

folder_path = 'Downloads/colored_DR_1000_each/severe'
image_files = os.listdir(folder_path)

# Show first 5 images
for i, img_name in enumerate(image_files[:5]):
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)
    
    # Convert BGR to RGB for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.title(f'Image {i+1}')
    plt.axis('off')
    plt.show()


# In[4]:


d=len(os.listdir('Downloads/colored_DR_1000_each/moderate'))
print(d)


# In[5]:


import os
import cv2
import matplotlib.pyplot as plt

folder_path = 'Downloads/colored_DR_1000_each/moderate'
image_files = os.listdir(folder_path)

# Show first 5 images
for i, img_name in enumerate(image_files[:5]):
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)
    
    # Convert BGR to RGB for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.title(f'Image {i+1}')
    plt.axis('off')
    plt.show()


# In[6]:


p=len(os.listdir('Downloads/colored_DR_1000_each/proliferate'))
print(p)


# In[7]:


import os
import cv2
import matplotlib.pyplot as plt

folder_path = 'Downloads/colored_DR_1000_each/proliferate'
image_files = os.listdir(folder_path)

# Show first 5 images
for i, img_name in enumerate(image_files[:5]):
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)
    
    # Convert BGR to RGB for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.title(f'Image {i+1}')
    plt.axis('off')
    plt.show()


# In[8]:


h=len(os.listdir('Downloads/colored_DR_1000_each/healthy'))
print(h)


# In[9]:


import os
import cv2
import matplotlib.pyplot as plt

folder_path = 'Downloads/colored_DR_1000_each/healthy'
image_files = os.listdir(folder_path)

# Show first 5 images
for i, img_name in enumerate(image_files[:5]):
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)
    
    # Convert BGR to RGB for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.title(f'Image {i+1}')
    plt.axis('off')
    plt.show()


# In[10]:


m=len(os.listdir('Downloads/colored_DR_1000_each/mild'))
print(m)


# In[11]:


import os
import cv2
import matplotlib.pyplot as plt

folder_path = 'Downloads/colored_DR_1000_each/mild'
image_files = os.listdir(folder_path)

# Show first 5 images
for i, img_name in enumerate(image_files[:5]):
    img_path = os.path.join(folder_path, img_name)
    img = cv2.imread(img_path)
    
    # Convert BGR to RGB for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.title(f'Image {i+1}')
    plt.axis('off')
    plt.show()


# In[12]:


dataset = []
label = []

INPUT_SIZE = 64

for i , image_name in enumerate(os.listdir("Downloads/colored_DR_1000_each/mild")):
    image = cv2.imread('Downloads/colored_DR_1000_each/mild/'+ image_name)
    image = Image.fromarray(image , 'RGB')
    image = image.resize((INPUT_SIZE , INPUT_SIZE)) # we are resizing images
    dataset.append(np.array(image))
    label.append(0)


#print(f'dataset of no :: {dataset} ')
print(len(dataset))
for i , image_name in enumerate(os.listdir('Downloads/colored_DR_1000_each/moderate')):
    image = cv2.imread('Downloads/colored_DR_1000_each/moderate/'+ image_name)
    image = Image.fromarray(image , 'RGB')
    image = image.resize((INPUT_SIZE , INPUT_SIZE)) # we are resizing images
    dataset.append(np.array(image))
    label.append(1)

print(len(dataset))
for i , image_name in enumerate(os.listdir('Downloads/colored_DR_1000_each/proliferate')):
    image = cv2.imread('Downloads/colored_DR_1000_each/proliferate/'+ image_name)
    image = Image.fromarray(image , 'RGB')
    image = image.resize((INPUT_SIZE , INPUT_SIZE)) # we are resizing images
    dataset.append(np.array(image))
    label.append(2)

print(len(dataset))
for i , image_name in enumerate(os.listdir('Downloads/colored_DR_1000_each/severe')):
    image = cv2.imread('Downloads/colored_DR_1000_each/severe/'+ image_name)
    image = Image.fromarray(image , 'RGB')
    image = image.resize((INPUT_SIZE , INPUT_SIZE)) # we are resizing images
    dataset.append(np.array(image))
    label.append(3)
    
    
for i , image_name in enumerate(os.listdir('Downloads/colored_DR_1000_each/healthy')):
    image = cv2.imread('Downloads/colored_DR_1000_each/healthy/'+ image_name)
    image = Image.fromarray(image , 'RGB')
    image = image.resize((INPUT_SIZE , INPUT_SIZE)) # we are resizing images
    dataset.append(np.array(image))
    label.append(4)
#print(dataset)
print(len(dataset))
print(len(label))

dataset = np.array(dataset)
label = np.array(label)


# Train test split

# In[13]:


x_train , x_test , y_train , y_test = train_test_split(dataset , label , test_size = 0.2 , train_size=0.8
                                                       , random_state=42, shuffle=True)
#this is actually x and y coordinate
print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

# divide data into 80% and 20%

x_train = normalize(x_train  , axis=1)
x_test = normalize(x_test  , axis=1)

y_train = to_categorical(y_train , num_classes=5)
y_test = to_categorical(y_test , num_classes=5)


# Model building

# In[14]:


from keras.activations import sigmoid
from tensorflow.keras.layers import Flatten
# from keras.backend import flatten
# from keras.engine.sequential import Sequential
#Model builing

model  = Sequential() # Initialising CNN
model.add(Conv2D(32 , (3,3) , input_shape = (INPUT_SIZE , INPUT_SIZE , 3))) # step -1 Convolution
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2 , 2))) # step -2  Pooling

#Adding second convolutional layer
model.add(Conv2D(32 , (3,3) , kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2 , 2)))

#Adding third convolutional layer

model.add(Conv2D(64 , (3,3) , kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2 , 2)))


model.add(Flatten()) # step -3 :  Flattening (hidden layer)
model.add(Dense(64)) #
model.add(Activation('relu')) # step - 4 : full connection
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax')) # step - 5 : output layer

#Binary CrossEntropy = 1 , sigmoid


# In[15]:


model.summary()


# In[16]:


# # Training the CNN (Compiler the cnn)
# model.compile(loss='categorical_crossentropy' , optimizer = 'adam' ,  metrics=['accuracy'])

# training the cnn on training dataset
# res = model.fit(np.array(x_train), np.array(y_train), verbose=1, epochs=20 ,shuffle = False)


# In[17]:


import tensorflow as tf

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

res = model.fit(
    np.array(x_train), np.array(y_train),
    epochs=15,
    batch_size=32,
    validation_split=0.1
    # validation_data=(X_val, y_val)  # Adding validation data
)


# In[18]:


from tensorflow.keras.optimizers import SGD

# Define the SGD optimizer (you can tweak learning rate, momentum, etc.)
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Compile the model with SGD
model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd,
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

# Fit the model
res = model.fit(
    np.array(x_train), np.array(y_train),
    epochs=15,
    batch_size=32,
    validation_split=0.1
)


# In[19]:


y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
len(y_true)


# In[20]:


print(x_test.shape)


# In[21]:


from sklearn.metrics import classification_report, confusion_matrix
# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))


# In[22]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Predict
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=True, linewidths=0.5)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[23]:


import matplotlib.pyplot as plt
# Plot training & validation accuracy
plt.plot(res.history['accuracy'], label='Training Accuracy')
plt.plot(res.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# In[32]:


import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
image = cv2.imread('Downloads/colored_DR_1000_each/severe/Severe_original_fe0fc67c7980.png_f46e189d-6f99-4c8b-846a-e3fc627eaf3e.png')
image = cv2.imread('Downloads/colored_DR_1000_each/proliferate/Proliferate_DR_original_ff8a0b45c789.png_cdc88b59-053d-4c72-bdda-1a478e10ee15.png')
image = cv2.imread('Downloads/colored_DR_1000_each/moderate/ffec9a18a3ce.png')
image = cv2.imread('Downloads/colored_DR_1000_each/mild/Mild_original_fecf4c5ae84b.png_569053c1-97f7-4a0a-ab8e-198132a7d159.png')
image = cv2.imread('Downloads/colored_DR_1000_each/healthy/No_DR_original_ffc04fed30e6.png_591d9248-b965-4be6-8cbd-0ec199429822.png')
img = Image.fromarray(image)
img = img.resize((64 , 64))
img = np.array(img)

print(img)

#result = model.predict_classes(img)
input_img = np.expand_dims(img , axis=0)
predict_x=model.predict(input_img)
result=np.argmax(predict_x,axis=1)
print(f'Now detect img is affected or not :: {result} ')



# In[ ]:





# In[ ]:




