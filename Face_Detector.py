import cv2 

#Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Capure video from webcam. 
webcam = cv2.VideoCapture(0) #VideoCapture(video path or 0 which is the defualt camera)

# Iterate forever over frames
while True:

    #read the current frame
    successful_frame_read, frame = webcam.read()

    #Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces 
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y + h), (0, 255, 0), 5)  #rectangle(image, coordinates of the face, color, thickness of the rectangle)

    #shows the image in a window
    cv2.imshow("Face Detector", frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

webcam.release()
