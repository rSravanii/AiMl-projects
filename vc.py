import cv2  #opencv
vs = cv2.VideoCapture(0) #initialize the camera

while True:
    _,img = vs.read() #reading the frame from the caamera

    cv2.imshow(" VideoStream ", img)
    # wait for 1 millisecond.The loop continues if no key is pressed.

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"): #if you press q here ,the camera will stop recording and
                              # windows will close 
           break
        
vs.release() #releasing the camera
cv2.destroyAllWindows() #close all windows 
