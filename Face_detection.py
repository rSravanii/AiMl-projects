import cv2  #openCV

#Lets load the pre-trained classifier for face detection which  already opencv has pre-trained models called HAAR CASCADES
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')


#or real time face detection we need to load the system camera

vc = cv2.VideoCapture(0) #where 0 is the default camera

#starting the main loop
while True:
    ret , frame = vc.read()
    if not ret:
        break
    
    #converting the image into grayscale image
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    #now we use clasifier to detect the faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1,  minNeighbors = 5)
    
    #now after detection , we have to make a frame for recognizing the face inthe frame like rectangle
    #(draw a rectangle around the face)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Face Detected", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Face in Frame", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    # Resize the frame to reduce window size
    frame = cv2.resize(frame, (800, 480))  # You can change this to (320, 240) for smaller view
         
    # Set font parameters
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 6
    color_detected = (0, 255, 0)
    color_none = (0, 0, 255)

    

    #noe to displat the results
    cv2.imshow('face detection ' , frame)

    #we needto press (q) to quit 
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

#here we release the camera and the window will be closed
vc.release()
cv2.destroyAllWindows()#enter q to release the camera
