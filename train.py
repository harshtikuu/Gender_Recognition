from sklearn.cross_validation import train_test_split
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
from voice import load_data
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import os,sys


#Preprocess the data
data=load_data()
x,y=data['data'],data['target']
x=StandardScaler().fit_transform(x)
X_train,X_test,Y_train,Y_test=train_test_split(x,y)



#Check if the model already exists


if os.path.exists('voicemodel.h5'):
	model=load_model('voicemodel.h5')
	print('\n')
	# Print accuracy on test data
	print('Test data accuracy = {} %'.format(model.evaluate(X_test,Y_test)[1]*100))

else:

#Build a model

	model=Sequential()
	model.add(Dense(35,activation='relu',input_dim=20))
	model.add(Dense(35,activation='relu'))
	model.add(Dropout(0.6))
	model.add(Dense(1,activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Train the model
	model.fit(X_train,Y_train,epochs=1000)
	model.save('voicemodel.h5')
	print('\n')
#Test the model on Test datasets	
	print('Test data accuracy = {} %'.format(model.evaluate(X_test,Y_test)[1]*100))
