from django.shortcuts import render

# Create your views here.
import cv2

from random import randrange

def detector(request):
    if 'q' in request.GET:
        trained_face_data = cv2.CascadeClassifier('C:\\Users\\USER\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\cv2\data\\haarcascade_frontalface_default.xml')
        webcam = cv2.VideoCapture(0)
        while True:
            successful_frame_read, frame = webcam.read()
            grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
            for (x,y,w,h) in face_coordinates:
                cv2.rectangle(frame, (x,y), ((x+w),(y+h)), (0,255,0), 2)
            cv2.imshow('Face Fetector',frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        webcam.release()
        cv2.destroyAllWindows()
        return render(request, 'index.html')
    else:
        
        return render(request, 'index.html')