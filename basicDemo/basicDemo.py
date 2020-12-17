import numpy as np
import scipy.linalg as sp
import matplotlib.pyplot as plt

# load the pixel data, resulting in a matlab matrix of dim 2000 x 240 called "mfeat_pix"
mfeat = open("mfeat-pix.txt", "r")
lines = mfeat.readlines()
pics = np.zeros((len(lines), 16, 15))
for i in range(0,len(lines)):
    lines[i] = [int(x) for x in lines[i].split()]
    pics[i] = np.reshape(lines[i],(16,15))

# plot the figure from the lecture notes.
fig, ax = plt.subplots(10, 10)
for i in range(0, 10):
    for j in range(0, 10):
        pic = 6 - pics[200*i + j]
        ax[i, j].imshow(pic, cmap='gray', vmin=0, vmax=6)
#plt.show()

# split the data into a training and a testing dataset
trainPatterns = np.zeros((1000, 240))
testPatterns = np.zeros((1000, 240))
for i in range(0, 20, 2):
    for j in range(0, 100):
        trainPatterns[i*50 + j] = lines[i*100 + j]
        testPatterns[i*50 + j] = lines[(i+1)*100 + j]
        
# create indicator matrices size 10 x 1000 with the class labels coded by 
# binary indicator vectors
b = np.ones((1,100));
trainLabels = sp.block_diag(b, b, b, b, b, b, b, b, b, b)
testLabels = trainLabels;
correctLabels = np.array([])
for i in range(0,10):
    correctLabels = np.concatenate((correctLabels, i*b), axis=None)
    
# from here, a demo implementation of a linear classifer based on 
# the ten class-mean features (hand-made features f_3 from the 
# lecture notes)

meanTrainImages = np.zeros((10,240));
for i in range(0, 10):
    meanTrainImages[i] = np.mean(trainPatterns[i*100:i*100 + 99], axis = 0);
    
featureValuesTrain = meanTrainImages @ trainPatterns.T;
featureValuesTest = meanTrainImages @ testPatterns.T;

# compute linear regression weights W
W = np.linalg.inv(featureValuesTrain @ featureValuesTrain.T) @ featureValuesTrain @ trainLabels.T

# compute train misclassification rate
classificationHypothesesTrain = W @ featureValuesTrain;
maxValues = np.max(classificationHypothesesTrain, axis=0);
maxIndicesTrain = np.where(classificationHypothesesTrain == maxValues)
nrOfMisclassificationsTrain = np.sum(correctLabels != maxIndicesTrain[0]);
print(maxIndicesTrain[0])
print(correctLabels)
print(correctLabels != maxIndicesTrain[0])
print("train misclassification rate = " + str(nrOfMisclassificationsTrain / 1000));

# compute test misclassification rate
classificationHypothesesTest = W @ featureValuesTest;
maxValues = np.max(classificationHypothesesTest, axis=0);
maxIndicesTest = np.where(classificationHypothesesTest == maxValues)
nrOfMisclassificationsTest = np.sum(correctLabels != maxIndicesTest[0]);
print("test misclassification rate = " + str(nrOfMisclassificationsTest / 1000));