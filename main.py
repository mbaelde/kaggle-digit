#%% Import relevant modules
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import keras             as K

#%% Load train data
folderData       = 'Data\\'
folderResults    = 'Results\\'
folderModels     = 'TrainedModels\\'

trainData        = pd.read_csv(folderData+'train.csv')
# Get image and label from trainData
labelTrain       = trainData['label']
imageTrain       = trainData.drop("label",axis=1)
# One hot encoding of the labels
labelTrainOneHot = K.utils.to_categorical(labelTrain)
#%% Choose some image to displays
nTrain           = imageTrain.shape[0]
nRow             = 4
nCol             = 7
idx              = np.random.choice(np.arange(nTrain), nRow*nCol)
# Plot somes images
plt.figure(figsize=(13,12))
for i in range(nRow * nCol):
    plt.subplot(nRow, nCol, i + 1)
    plt.imshow(imageTrain.values[idx[i],:].reshape(28,28),cmap='gray')
    title_text = 'Image ' + str(i + 1) + ' labeled ' + str(labelTrain[idx[i]])
    plt.title(title_text, size=6.5)
    plt.xticks(())
    plt.yticks(())
plt.show()

#%% Setup a DNN
# Get input dimension
dInput          = imageTrain.shape[1]
# Number of units for the hidden layer
nUnits          = 1000
# Number of classes to predict
nClasses        = 10

# Construct the DNN : one input layer, one hidden layer and one output layer
inputLayer      = K.layers.Input((dInput,),name='inputLayer')
denseLayer_1    = K.layers.Dense(nUnits,activation='relu')(inputLayer)
denseLayer_2    = K.layers.Dense(nClasses,activation='relu')(denseLayer_1)
# Add softmax layer for class probability computation
outputLayer     = K.layers.Softmax()(denseLayer_2)
# Gather the layers into a keras model
model           = K.Model(inputs=inputLayer, outputs=outputLayer)
# Compile the model with ADAM optimizer and binary crossentropy loss
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
# Fit the DNN model by batch of 128 samples and for 10 epochs
model.fit(imageTrain,labelTrainOneHot,batch_size=128,epochs=10)
# Save the trained model
model.save('TrainedModel\\myDNN.h5')

#%% Lood test data
testData        = pd.read_csv(folderData+'test.csv')
# Get image from testData
imageTest       = testData
#%% Predict on test data
labelPredicted  = model.predict(imageTest)