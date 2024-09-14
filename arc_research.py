import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from threading import Thread


class ThreadedCamera:
    def __init__(self, source = 0):

        self.capture = cv2.VideoCapture(source)

        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame  = None

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def grab_frame(self):
        if self.status:
            return self.frame
        return None  



# Initialize the ArcFace model
model = insightface.model_zoo.get_model('arcface_r100_v1')
# model.prepare(ctx_id=0)  # Use CPU. For GPU, use ctx_id=0

# Initialize OpenCV for face detection
cap = cv2.VideoCapture("video.mp4")  # Use webcam or video file

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Initialize face detection app from InsightFace
face_app = FaceAnalysis()
face_app.prepare(ctx_id=-1, det_size=(640, 640))  # Adjust detection size

# ------------------------------------------------------------------------------------------------------

# while True:
#     # Read a frame from the video
#     ret, frame = cap.read()

#     # If frame was not read correctly, exit
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break

#     # Detect faces in the frame
#     faces = face_app.get(frame)

#     for face in faces:
#         # Draw a rectangle around the detected face
#         bbox = face.bbox.astype(int)
#         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

#         # Crop the face and get embeddings
#         face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#         face_embedding = face.normed_embedding
#         print("Face Embedding:", face_embedding)

#     # Show the frame with the detected face
#     cv2.imshow('ArcFace Stream', frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and close the windows
# cap.release()
# cv2.destroyAllWindows()





# -----------------------------------------------------------------------------------------------------------
import time
streamer = ThreadedCamera("video.mp4")
while True:
    frame = streamer.grab_frame()
    
    if frame is None:
        print("frame error")
        print(streamer.status)
        time.sleep(1)
        continue
   
    faces = face_app.get(frame)

    for face in faces:
        # Draw a rectangle around the detected face
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        # Crop the face and get embeddings
        face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # face_embedding = face.normed_embedding
        # print("Face Embedding:", face_embedding)
        
    # Show the frame with the detected face
    cv2.imshow('ArcFace Stream', frame)


    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
streamer.capture.release()
# streamer.release()
cv2.destroyAllWindows()
