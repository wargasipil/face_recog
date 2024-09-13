import cv2
import torch
from retinaface import RetinaFace

model = RetinaFace(quality='normal')  # Use 'best' for highest accuracy

def detect_faces(frame):
    # Convert the frame to RGB (RetinaFace expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = model.predict(rgb_frame)

    return results

def main(file_path):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Break the loop if no frame is retrieved (end of video)
        if not ret:
            print("End of video.")
            break
        
        # Detect faces
        faces = detect_faces(frame)

        # Draw bounding boxes on detected faces
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {face['score']:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        # Display the frame
        cv2.imshow('Video Stream', frame)

        # Exit the video stream if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to your MP4 file
    video_file_path = './video.mp4'
    
    main(video_file_path)