#%% Import relevant modules
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

#%% 
folderData    = 'data\\'
folderResults = 'results\\'

trainData = pd.read_csv(folderData+'train.csv')

labelData = trainData.iloc[:,0]
imageData = trainData.iloc[:,1:]
#%%