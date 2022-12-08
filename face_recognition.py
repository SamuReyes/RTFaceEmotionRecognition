import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from keras.models import load_model
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#load model
model = load_model('model_rgb_1.h5')

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, color = image.shape
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.detections:
      # Get current faces
      for faces in range(len(results.detections)):

        detection = results.detections[faces]
        #mp_drawing.draw_detection(image, detection)


        location = detection.location_data

        relative_bounding_box = location.relative_bounding_box
        rect_start_point = _normalized_to_pixel_coordinates(
          relative_bounding_box.xmin, relative_bounding_box.ymin, width,
          height)
        rect_end_point = _normalized_to_pixel_coordinates(
          relative_bounding_box.xmin + relative_bounding_box.width,
          relative_bounding_box.ymin + relative_bounding_box.height, width,
          height)
        if (rect_start_point is not None) and (rect_end_point is not None):
          xleft, ytop = rect_start_point
          xright, ybot = rect_end_point

          # Crop the face from the frame
          crop_img = image[ytop: ybot, xleft: xright]
          # Resize the face to 48x48px for classification neural network
          crop_img = cv2.resize(crop_img, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)

          # Change dimensionality of image to use as valid input of the model
          crop_img = crop_img[np.newaxis, :]

          # Get model output
          predictions = model.predict(crop_img)

          # Find max indexed array
          max_index = np.argmax(predictions[0])

          # Get emotion from model output
          emotions = ('angry', 'happy','neutral', 'sad', 'surprise')
          predicted_emotion = emotions[max_index]

          # Put emotion text on image
          text_x, text_y = rect_start_point
          cv2.putText(image, predicted_emotion, (text_x,text_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

          ## Lets draw a bounding box
          color = (255, 0, 0)
          thickness = 2
          cv2.rectangle(image, rect_start_point, rect_end_point, color, thickness)
          xleft, ytop = rect_start_point
          xright, ybot = rect_end_point

          cv2.imshow('Face Recognition Analysis', image)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
      break

cap.release()
cv2.destroyAllWindows
