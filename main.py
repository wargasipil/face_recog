import cv2
import numpy as np
import insightface
from faceindex import FaceIndex


def initialize_insightface_model():
    model = insightface.app.FaceAnalysis()  # Initialize face analysis model
    model.prepare(ctx_id=-1)  # Use GPU (ctx_id=0), or CPU (ctx_id=-1)
    return model



# --------------------------------------
model = initialize_insightface_model()
index = FaceIndex()

# Open video capture
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Check the actual FPS
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame rate: {fps}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    faces = model.get(frame)

    face_count = 0
    for face in faces:
        face_count += 1

        # getting embedding
        embedding = face.normed_embedding  # Get embedding for the first detected face
        embedding = np.array([embedding])

        # search embeding
        distances, indices, labels = index.search(embedding)

        bbox = face.bbox.astype(int)
        topleft = (bbox[0], bbox[1])
        cv2.rectangle(frame, topleft, (bbox[2], bbox[3]), (255, 0, 0), 2)

        # Add the label
        
        if len(labels) > 0:
            label_position = (topleft[0], topleft[1] - 10)
            cv2.putText(frame, labels[0], label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()