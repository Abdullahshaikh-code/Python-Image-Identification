from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib
import numpy as np

with_mask=np.load("with_mask.npy").reshape(201,50*50*3)
without_mask=np.load("without_mask.npy").reshape(201,50*50*3)
X=np.r_[with_mask,without_mask]
labels=np.zeros(X.shape[0])
labels[200:]=1.0

xTrain,xTest,yTrain,yTest=train_test_split(X,labels,test_size=0.25)
pca=PCA(n_components=3)
xTrain=pca.fit_transform(xTrain)
xTest=pca.fit_transform(xTest)

svm=SVC()
model=svm.fit(xTrain,yTrain)
yPrediction=model.predict(xTest)
print(accuracy_score(yTest,yPrediction))

joblib.dump(model, 'Mask_detection_model.pkl')
joblib.dump(pca, 'pca_transform.pkl')