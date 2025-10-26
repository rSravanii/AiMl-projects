import cv2     
import imutils  #fir resize the image
import time #delay
import datetime

VIDEO_SOURCE = 0  # Set the video source (0 for webcam, or a path to a video file)

MIN_AREA = 500  # Minimum area (in pixels^2) for a contour to be considered movement

vs = cv2.VideoCapture(VIDEO_SOURCE)  # Initializing: Start the video stream (can be a webcam or video file)

print("[INFO] Warming up camera...")  # Give the camera a moment to warm up
time.sleep(2.0)  #for detection

firstFrame = None  # Initialize the first frame to None. This will be the reference frame (background).
previous_status_text = ""
moving_detected = True

print("[INFO] Starting motion detection...")

while True:     #this is the main loop
    
    ret, frame = vs.read() ## Grab the current frame

    if not ret:      # Check if a frame was successfully read
        print("[INFO] No frame read, exiting...")
        break

    # 1. Pre-process the frame
    
    frame = imutils.resize(frame, width=500) # Resize the frame for faster processing (optional, but recommended)
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     # Convert to grayscale

    gray = cv2.GaussianBlur(gray, (21, 21), 0)      # Apply Gaussian blur to smooth out noise and detail

    # 2. Initialize the background reference frame
    
    if firstFrame is None:   # If the first frame is None, initialize it
        firstFrame = gray
        print("[INFO] [TERMINAL STATUS ] Background establieshed . Ready for motion detection ")
        continue     # Continue to the next loop iteration

    # 3. Calculate the difference (Motion detection core)
    
    frameDelta = cv2.absdiff(firstFrame, gray)  # Compute the absolute difference between the current frame and the reference frame
    
    # Apply a threshold to reveal areas of movement (white pixels)
    # Pixels with a difference less than 25 are set to black (0); others are set to white (255)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # 4. Dilate and Find Contours
    
    thresh = cv2.dilate(thresh, None, iterations=2)  # Dilate the thresholded image (filling in small holes)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,     # Find contours (outlines) in the thresholded image
        cv2.CHAIN_APPROX_SIMPLE)
   
    cnts = imutils.grab_contours(cnts)  # Get the contours from the imutils output

    moving_detected = False # Reset detection status for the current frame

    # 5. Loop over the contours
    
    for c in cnts:
        # If the contour is too small, ignore it
        if cv2.contourArea(c) < MIN_AREA:
            continue

        moving_detected = True  # A significant movement has been detected!

        (x, y, w, h) = cv2.boundingRect(c)    # Compute the bounding box for the contour
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw the bounding box on the original color frame

    # 6. Display Status Text
    if moving_detected:
        current_status_text = "Moving Object Detected.."
        color = (0, 0, 255) # Red for movement
    else:
        current_status_text = "Normal"
        color = (0, 255, 0) # Green for no movement
    if current_status_text !=previous_status_text:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [TERMINAL STATUS] {current_status_text}")
        previous_status_text = current_status_text

    cv2.putText(frame, "Status: {}".format(current_status_text), (10, 20),     # Put the status text on the frame
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
     # Put the current time on the frame
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # 7. Show the processed frames (optional: uncomment to see internal steps)
    #cv2.imshow("Thresh", thresh)
    #cv2.imshow("Frame Delta", frameDelta)
    
    cv2.imshow("Security Feed", frame)  # Show the main result frame

    # 8. Handle exit
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break
    
vs.release()   # Release the video capture object

cv2.destroyAllWindows()  # Close all OpenCV windows

