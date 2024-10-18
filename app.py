from flask import Flask, request, jsonify
import torch
from PIL import Image
import io

# Initialize Flask application
app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image'].read()
    img = Image.open(io.BytesIO(image_file))

    # Perform inference
    results = model(img)

    # Extract the data from the results
    results_json = results.pandas().xyxy[0].to_json(orient="records")

    return jsonify({"detections": results_json}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
