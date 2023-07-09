from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import prooces

app = Flask(__name__)
CORS(app)


@app.route('/process', methods=['POST'])
def process_image():
    image_file = request.files['image']
    my_string = request.form['string']
    # Read the string data
    # print(my_string)
    # Read the image using OpenCV
    img = cv2.imdecode(np.fromstring(
        image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform image processing using OpenCV
    processed_image = process_image_with_opencv(img, my_string)

    # Convert the processed image to Base64
    processed_image_data = convert_to_base64(processed_image)

    # Return the Base64 encoded image data as response
    return jsonify({'image_data': processed_image_data})


def process_image_with_opencv(img, my_string):
    # Perform your image processing operations using OpenCV
    # Use the my_string data in your processing logic
    # ...

    gray = prooces.process_ing(img, my_string)

    # You can perform additional operations on the processed image
    # ...

    # Return the processed image
    return gray


def convert_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    encoded_string = base64.b64encode(buffer)
    return encoded_string.decode('utf-8')


if __name__ == '__main__':
    app.run()
