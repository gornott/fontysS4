# import cv2
# face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# video = cv2.VideoCapture(1)
# while True:
#     gray = cv2.cvtColor(video.read()[1], cv2.COLOR_BGR2GRAY)
#     check, frame = video.read()
#     faces = face_data.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
#     cv2.imshow('frame',frame)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()



