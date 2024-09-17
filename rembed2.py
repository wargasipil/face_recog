import cv2  # For webcam capture
import insightface
import numpy as np
import faiss
import time

# Function to initialize the InsightFace model for face detection and feature extraction
def initialize_insightface_model():
    model = insightface.app.FaceAnalysis()  # Initialize face analysis model
    model.prepare(ctx_id=-1)  # Use GPU (ctx_id=0), or CPU (ctx_id=-1)
    return model

# Function to capture a frame from the webcam
def capture_frame_from_webcam():
    # Start video capture (0 is typically the default webcam)
    cap = cv2.VideoCapture(0)

    print("Press 's' to save a face or 'q' to quit")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture image from webcam.")
            break

        # Display the frame
        cv2.imshow('Webcam - Press s to Save Face, q to Quit', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        time.sleep(0.01)

        # If 's' is pressed, save the current frame
        if key == ord('s'):
            print("Face captured!")
            break
        # If 'q' is pressed, quit
        elif key == ord('q'):
            print("Exiting without capturing.")
            frame = None
            break

    # Release the capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()

    return frame

# Function to get the face embedding from a webcam frame
def get_face_embedding_from_frame(model, frame):
    # Detect and extract face embeddings from the captured frame
    faces = model.get(frame)

    if len(faces) > 0:
        embedding = faces[0].normed_embedding  # Get embedding for the first detected face
        return np.array([embedding])  # Convert to a NumPy array for FAISS
    else:
        print("No face detected in the captured frame.")
        return None

# Function to initialize the FAISS index (Flat L2 distance)
def initialize_faiss_index(embedding_dim):
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance-based index
    return index

# Function to add embeddings to the FAISS index
def add_embedding_to_faiss(index, embedding):
    index.add(embedding)  # Add the embedding(s) to FAISS index

# Function to search for similar faces in FAISS index
def search_faiss(index, query_embedding, k=5):
    distances, indices = index.search(query_embedding, k)  # Search for k nearest neighbors
    return distances, indices

# Function to save the FAISS index to a file
def save_faiss_index(index, filename="face_embedding.index"):
    faiss.write_index(index, filename)

# Function to load the FAISS index from a file
def load_faiss_index(filename="face_embedding.index"):
    index = faiss.read_index(filename)
    return index

# Main function for registration and face recognition
def main():
    # Initialize InsightFace model
    model = initialize_insightface_model()

    # Initialize FAISS index (embedding size is typically 512 for InsightFace)
    embedding_dim = 512
    index = initialize_faiss_index(embedding_dim)

    # Step 1: Register faces via webcam
    while True:
        # Capture frame from webcam
        frame = capture_frame_from_webcam()
        
        if frame is None:
            break

        # Get the embedding for the captured face
        embedding = get_face_embedding_from_frame(model, frame)
        
        if embedding is not None:
            # Add the embedding to FAISS index
            add_embedding_to_faiss(index, embedding)
            print("Face embedding registered successfully!")

        # Ask if the user wants to register another face
        cont = input("Do you want to register another face? (y/n): ")
        if cont.lower() != 'y':
            break

    # Save the FAISS index
    save_faiss_index(index)
    print("FAISS index saved.")

    # Step 2: Detect and recognize a face from FAISS index
    print("\n---- Face Recognition ----")
    
    # Capture a query face for recognition
    print("Capture a face for recognition.")
    query_frame = capture_frame_from_webcam()

    if query_frame is not None:
        # Extract embedding for the query face
        query_embedding = get_face_embedding_from_frame(model, query_frame)

        if query_embedding is not None:
            # Load FAISS index (if needed)
            index = load_faiss_index()

            # Search for similar faces in the FAISS index
            k = 5  # Find top 5 matches
            distances, indices = search_faiss(index, query_embedding, k)

            # Output the results
            print(f"\nSearch Results (Top {k} Matches):")
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                print(f"Match {i+1}: Index {idx}, Distance {dist}")

            # You can implement your own threshold for a "match"
            # E.g., if the distance is below a certain threshold, it's considered a match
            threshold = 1.0  # Adjust this threshold based on testing
            if distances[0][0] < threshold:
                print(f"\nFace recognized! Closest match at index {indices[0][0]} with distance {distances[0][0]}")
            else:
                print("\nNo match found within the threshold.")

if __name__ == "__main__":
    main()
