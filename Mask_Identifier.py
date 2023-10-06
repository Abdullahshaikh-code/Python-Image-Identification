import cv2
import joblib

model = joblib.load('Mask_detection_model.pkl')
pca=joblib.load("pca_transform.pkl")
haarData=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

label={
        0:"Mask Detected",
       1:"Mask Not Detected"
       }
condition=True
while condition:
        img=cv2.imread("TestData/img.jpeg")
        faces=haarData.detectMultiScale(img)
        for x,y,width,height in faces:
                cv2.rectangle(img,(x,y),(x+width,y+height),(270,190,200),3)
                face = img[y:y+height, x:x+width]
                face=cv2.resize(face,(50,50)).reshape(1,-1)
                face=pca.transform(face)
                predition=model.predict(face)
                text=(label[int(predition)])
                print(text)
                cv2.putText(img, text, (x-100, y + height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 55, 255), 2)

        cv2.imshow("Image",img)
        if cv2.waitKey(2)==27 :
                break
cv2.destroyAllWindows()

