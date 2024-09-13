import cv2

# Open video capture (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Check if the video capture is initialized correctly
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Break the loop if the frame is not successfully captured
    if not ret:
        print("Error: Could not read frame.")
        break


    # Display the resulting frame
    cv2.imshow('Video Stream', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
