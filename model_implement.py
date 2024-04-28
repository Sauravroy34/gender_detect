import cv2
import numpy as np
from tensorflow.keras.models import load_model
gen_cnn = load_model("gender_predict.keras")
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX




cap = cv2.VideoCapture(0)  
font = cv2.FONT_HERSHEY_SIMPLEX 

while cap.isOpened():
    ret, frame = cap.read()  
    if not ret:
        print("Error: Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4)

    new_frame = cv2.resize(frame, (150, 150)) 
    arr = np.array([new_frame])  

   
    predictv = gen_cnn.predict(arr)  


    predicted_class = np.argmax(predictv)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=3)

      
        text_x = x + int(w / 2) - 25  
        text_y = y + h + 15  
        text = "Male" if predicted_class == 1 else "Female"  
        cv2.putText(frame, text, (text_x, text_y), font, 0.7, (0, 255, 0), 2)

       
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

    cv2.imshow("window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()

