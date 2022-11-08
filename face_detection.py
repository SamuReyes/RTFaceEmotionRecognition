import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

i = 0

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

    # Iterator over frames
    i += 1
    i %= 30

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
      if i == 0:
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
            # Save the face in a jpg file
            cv2.imwrite('crop'+str(faces)+'.jpg', crop_img)

            ## Lets draw a bounding box
            color = (255, 0, 0)
            thickness = 2
            cv2.rectangle(image, rect_start_point, rect_end_point, color, thickness)
            xleft, ytop = rect_start_point
            xright, ybot = rect_end_point

      # Draw the bounding box
      else:
        for faces in range(len(results.detections)):
          detection = results.detections[faces]

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

            ## Lets draw a bounding box
            color = (255, 0, 0)
            thickness = 2
            cv2.rectangle(image, rect_start_point, rect_end_point, color, thickness)
            xleft, ytop = rect_start_point
            xright, ybot = rect_end_point

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Emotions', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
