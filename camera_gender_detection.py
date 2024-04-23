import cv2
from deepface import DeepFace

# Open the default camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Use try-except to handle exceptions raised when no face is detected
    try:
        # Analyzing gender with enforce_detection set to True for stricter face detection
        results = DeepFace.analyze(frame, actions=['gender'], enforce_detection=True)
        
        # Adjust access to the results since it's a list
        if len(results) > 0:
            result = results[0]  # assuming the first result is the primary one to consider
            print("Person detected: Gender -", result['dominant_gender'])
            gender_details = result['gender']
            print("Detailed gender probabilities: Woman - {:.2f}%, Man - {:.2f}%".format(gender_details['Woman'], gender_details['Man']))
        else:
            print("No results found.")

    except ValueError as ve:
        print("No face detected:", ve)
    except Exception as e:
        print("Error during detection:", e)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
