import cv2
import numpy as np
haarData=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
imgCapture=cv2.VideoCapture(0)
facesData=[]
names=["With_mask.npy"]
for i in range(2):
    condition=True
    while condition:
        status,img=imgCapture.read()
        if status:
            faces=haarData.detectMultiScale(img)
            for x,y,width,height in faces:
                cv2.rectangle(img,(x,y),(x+width,y+height),(270,190,200),3)
                face = img[y:y+height, x:x+width]
                face=cv2.resize(face,(50,50))
                print( len(facesData))
                if len(facesData)<=200:
                    facesData.append(face)
            cv2.imshow("Image",img)
            if cv2.waitKey(2)==27 or len(facesData)>200:
                break
    np.save(names[i],facesData)
imgCapture.release()
cv2.destroyAllWindows()

