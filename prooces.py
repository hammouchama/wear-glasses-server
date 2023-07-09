import io
import cv2
import dlib
import numpy as np
import urllib.request

# Load the pre-trained face detector from dlib
face_detector = dlib.get_frontal_face_detector()

# Load the landmark predictor from dlib
landmark_predictor = dlib.shape_predictor(
    'shape_predictor_68_face_landmarks.dat')

# Load the glasses image with alpha channel (transparency)

# Function to overlay glasses on the face


def overlay_glasses(face_img, glasses_img, landmarks):

    glasses_width = int(np.linalg.norm(landmarks[17] - landmarks[26]) * 1.2)
    glasses_height = int(
        glasses_width * glasses_img.shape[0] / glasses_img.shape[1])

    # Resize glasses image to match the calculated size
    glasses_resized = cv2.resize(glasses_img, (glasses_width, glasses_height))

    # Calculate the position to overlay glasses
    x = landmarks[17, 0] - int(glasses_width / 12)
    y = landmarks[20, 1] - int(glasses_height / 9)

    # Adjust the alpha channel of glasses image
    alpha = glasses_resized[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    # Calculate the region of interest (ROI) for the glasses
    roi = face_img[y:y + glasses_height, x:x + glasses_width]

    # Blend the glasses image with the ROI
    for c in range(0, 3):
        face_img[y:y + glasses_height, x:x + glasses_width, c] = (
            alpha * glasses_resized[:, :, c] + alpha_inv * roi[:, :, c])

    return face_img


def process_ing(image, glasses):

    frame = image
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Download the image from the URL
    image_data = urllib.request.urlopen(glasses).read()

    # Create an OpenCV image from the downloaded image data
    image = cv2.imdecode(np.asarray(bytearray(image_data),
                                    dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    # Save the image using OpenCV's imwrite
    cv2.imwrite("glasses.png", image)

    # image = io.imread(glasses)
    glasses_img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
    # Detect faces in the frame
    faces = face_detector(gray)

    # Process each face
    for face in faces:
        # Detect landmarks in the face
        landmarks = landmark_predictor(gray, face)

        # Convert dlib landmarks to numpy array
        landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Overlay glasses on the face
        frame = overlay_glasses(frame, glasses_img, landmarks_np)
    return frame
