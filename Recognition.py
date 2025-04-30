import cv2
import numpy as np
import face_recognition
import os
import dlib                                    
from scipy.spatial import distance as dist    
from imutils import face_utils     
from datetime import datetime
import pandas as pd

cap = cv2.VideoCapture(0)


cap.set(3,700)
cap.set(4,1020)


path = "photosss"
images =[]
className = []

mylist = os.listdir(path)

print(mylist)

for cl in mylist:
    curImg = cv2.imread("photosss/"+cl)
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])


print(len(images))

def findencodes(image):
    encodelist = []
    print(len(image))

    for img in image:
    
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)


    return encodelist    

# FOR CSV FILE


# def attendace(name):
#     with open("attendance.csv","r+") as mark:
#         datalist = mark.readlines()
#         namelist =[]

#         for line in datalist:
#             sp = line.split(",")
#             namelist.append(sp[0])

#         if name not in namelist:
#             now = datetime.now()

#             time = now.strftime('%H:%M:%S')    
#             mark.writelines(f'\n{name},{time}')




def attendace(name):
    file = 'attendance.xlsx'
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    if os.path.exists(file):
        try:
            df = pd.read_excel(file, engine='openpyxl')
        except Exception as e:
            print("Corrupted file detected, creating a new Excel file.")
            df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
    else:
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])

 
    already_marked = ((df['Name'] == name) & (df['Date'] == date_str)).any()

    if not already_marked:
        new_entry = pd.DataFrame([[name, date_str, time_str]], columns=['Name', 'Date', 'Time'])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_excel(file, index=False, engine='openpyxl')





encodedimages = findencodes(images)
print("Encoding Complatedd")

cap = cv2.VideoCapture(0)


cap.set(3,700)
cap.set(4,1020)
def calculate_EAR(eye) :

    y1 = dist.euclidean(eye[1] , eye[5])
    y2 = dist.euclidean(eye[2] , eye[4])

    x1 = dist.euclidean(eye[0],eye[3])

    EAR = (y1+y2) / x1
    return EAR



blink_thresh = 0.5
succ_frame = 2
count_frame = 0


(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']


detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
blink_counter = {}


while True:
    suc,img = cap.read()
    suc = img.copy()
    
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)

    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    curframe = face_recognition.face_locations(imgS)
    encodecru = face_recognition.face_encodings(imgS,curframe)


    for encodeface,faceloc in zip(encodecru,curframe):

        matches = face_recognition.compare_faces(encodedimages,encodeface)
        facedis = face_recognition.face_distance(encodedimages,encodeface)


        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        
            rect = dlib.rectangle(x1, y1, x2, y2)
            shape = landmark_predict(img, rect)
            shape = face_utils.shape_to_np(shape)

            lefteye = shape[L_start:L_end]
            righteye = shape[R_start:R_end]


            if name not in blink_counter:
               blink_counter[name] = 0


            left_EAR = calculate_EAR(lefteye)
            right_EAR = calculate_EAR(righteye)
            avg = (left_EAR + right_EAR) / 2

            if avg < blink_thresh:
                blink_counter[name] += 1
            else:
                if blink_counter[name] >= succ_frame:
                    cv2.putText(img, 'Blink Detected', (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                    attendace(name)
                blink_counter[name] = 0

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Face_reco",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break     
        
    

