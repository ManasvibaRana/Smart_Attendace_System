import cv2
import numpy as np
import face_recognition


img = face_recognition.load_image_file("photosss/Elon_musk.jpg")

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(img)[0]
faceframe = face_recognition.face_encodings(img)[0]


cv2.rectangle(img,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,0),2)

cv2.imshow("Elon",img)
cv2.waitKey(0)