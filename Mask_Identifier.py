from sklearn.decomposition import PCA
import cv2
import joblib

model = joblib.load('Mask_detection_model.pkl')
pca=joblib.load("pca_transform.pkl")
haarData=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

imgCapture=cv2.VideoCapture(0)
label={0:"Mask",1:"NoMask"}
condition=True
while condition:
        status,img=imgCapture.read()
        if status:
            faces=haarData.detectMultiScale(img)
            for x,y,width,height in faces:
                cv2.rectangle(img,(x,y),(x+width,y+height),(270,190,200),3)
                face = img[y:y+height, x:x+width]
                face=cv2.resize(face,(50,50)).reshape(1,-1)
                face=pca.transform(face)
                predition=model.predict(face)
                print(label[int(predition)])
            cv2.imshow("Image",img)
            if cv2.waitKey(2)==27 :
                break
imgCapture.release()
cv2.destroyAllWindows()

