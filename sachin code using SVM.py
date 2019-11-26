#!/usr/bin/env python
# coding: utf-8

# In[3]:



from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[4]:


cd C:\Users\sachin\Downloads


# In[5]:


data = pd.read_csv("Nome_Gold_reserve_dataset.csv")


# In[6]:


data


# In[7]:


data=data.iloc[:,1:5]


# In[8]:


data["Y"]


# In[9]:


X=data.iloc[:,0:-1]
Y=data.iloc[:,3]


# In[10]:


X


# In[11]:


#splitting the data into traning and testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[ ]:



# train_data = train_data[497 + 3]
# train_data['Target'] = target

# C_mat = train_data.corr()
# fig = plt.figure(figsize = (15,15))

# sb.heatmap(C_mat, vmax = .8, square = True)
# plt.show()


# In[12]:


from sklearn.svm import SVR
import numpy as np

clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
clf.fit(X_train, y_train) 


# In[14]:


y_pred=clf.predict(X_test)


# In[15]:


# finding root mean square error
from sklearn.metrics import mean_squared_error

from math import sqrt

rmse = sqrt(mean_squared_error(y_test, y_pred))

print(rmse)


# In[17]:


# finding root mean square error
mse=mean_squared_error(y_test, y_pred)
print(mse)


# In[13]:


#R^2 value
clf.score(X_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[88]:


NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
NN_model.summary()


# In[89]:


history=NN_model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)


# In[94]:


# print(history.history.keys())
# # "Loss"
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()


# In[85]:


y_pred=NN_model.predict(X_test)


# In[86]:


from sklearn.metrics import accuracy_score
test_acc = NN_model.evaluate(X_train,y_train, verbose=0)


# In[93]:


print(NN_model.metrics_names)
test_acc


# In[ ]:




