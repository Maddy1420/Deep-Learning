# --------------> IMPORT LIBRARIES
import cv2
import PIL
import urllib.request
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Access MediaPipe solutions
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils


# -----------> HELPER FUNCTION TO FETCH IMAGE FROM URL
def url_to_array(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        image_bytes = np.asarray(bytearray(response.read()), dtype="uint8")
        image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        # Convert BGR (OpenCV default) to RGB for MediaPipe & Matplotlib
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


# -----------> LOAD IMAGE
# Use a verified sample mug image URL from Wikimedia
mug_url = "https://goodstock.photos/wp-content/uploads/2018/01/Laptop-Coffee-Mug-on-Table.jpg"

try:
    mug = url_to_array(mug_url)
except Exception as e:
    print(f"Failed to download image: {e}. Falling back to local 'cup.jpg'")
    image_bgr = cv2.imread('cup.jpg')
    if image_bgr is None:
        raise FileNotFoundError("Could not find 'cup.jpg' locally or fetch the web image.")
    mug = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# ------> INSTANTIATE OBJECTRON & RUN INFERENCE
with mp_objectron.Objectron(
    static_image_mode=True,
    max_num_objects=5,
    min_detection_confidence=0.1,  # Set lower threshold for better detection sensitivity
    model_name='Cup'
) as objectron:

    # Run inference on RGB image
    results = objectron.process(mug)

    # --------------> DISPLAY & DRAW RESULTS
    annotated_image = mug.copy()

    if not results.detected_objects:
        print('No 3D box landmarks detected.')
    else:
        print(f"Detected {len(results.detected_objects)} object(s).")
        for detected_object in results.detected_objects:
            # Draw 2D landmarks for 3D bounding box
            mp_drawing.draw_landmarks(
                annotated_image,
                detected_object.landmarks_2d,
                mp_objectron.BOX_CONNECTIONS
            )

            # Draw 3D axis based on rotation and translation
            mp_drawing.draw_axis(
                annotated_image,
                detected_object.rotation,
                detected_object.translation
            )

# -----------> PLOT THE RESULT
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(annotated_image)
ax.axis('off')
plt.show()