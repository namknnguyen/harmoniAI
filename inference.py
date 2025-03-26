import mediapipe as mp
import cv2
import numpy as np
import pickle
import os

class EmotionClassifier:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.model_path = 'emotion_model.pkl'
        self.model = self.load_model()

    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model file not found. Please train the model first.")
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)

    def run_inference(self):
        """Run real-time emotion classification"""
        # Try different camera indices
        for camera_index in [0, 1, -1]:
            print(f"Trying to open camera {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Successfully opened camera {camera_index}")
                break
            cap.release()
        else:
            print("Error: Could not open any camera")
            return

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Process the image
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks
                self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION)
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                                            for landmark in pose]).flatten())

                    # Extract Face landmarks
                    face = results.face_landmarks.landmark
                    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                                            for landmark in face]).flatten())

                    # Combine rows
                    row = pose_row + face_row
                    
                    # Make prediction
                    prediction = self.model.predict([row])[0]
                    
                    # Draw prediction on frame
                    cv2.putText(image, f'Emotion: {prediction}', (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                except Exception as e:
                    pass

                cv2.imshow('Emotion Classification', image)

                # Press 'q' to quit
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # This helps prevent window hanging

def main():
    try:
        classifier = EmotionClassifier()
        print("Starting emotion classification... Press 'q' to quit")
        classifier.run_inference()
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()