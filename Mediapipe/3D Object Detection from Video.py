import cv2
import mediapipe as mp

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

video_path = r"C:\Users\aarun\Agentic AI\DeepLearning\Mediapipe\shoe.mp4"
cap = cv2.VideoCapture(video_path)

objectron = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=5,
    min_detection_confidence=0.3,  # Lowered slightly for smoother tracking in video
    min_tracking_confidence=0.5,
    model_name='Shoe'
)

while cap.isOpened():
    success, image = cap.read()
    
    # End of video stream
    if not success:
        print("Video processing finished or cannot read frame.")
        break

    # Convert color space for MediaPipe processing
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = objectron.process(image)

    # Convert back for OpenCV display
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Render 3D bounding box and coordinate axes
    if results.detected_objects:
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(
                image,
                detected_object.landmarks_2d,
                mp_objectron.BOX_CONNECTIONS
            )
            mp_drawing.draw_axis(
                image,
                detected_object.rotation,
                detected_object.translation
            )

    # Do not mirror/flip video files (cv2.flip is only for live webcams)
    cv2.imshow('MediaPipe Objectron - Video', image)
    
    # Press 'q' to exit early or adjust frame delay (e.g., 25ms for ~40fps playback)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()