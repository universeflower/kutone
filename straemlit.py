import cv2
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

class ObjectDetector(VideoTransformerBase):
    def __init__(self, model):
        self.model = model

    def transform(self, frame):
        # Convert the frame to BGR format
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv8 inference on the frame
        results = self.model(img)

        # Visualize the results on the frame
        for label, confidence, bbox in zip(results.names, results.scores, results.xyxy):
            # Convert coordinates from normalized values to pixel values
            x1, y1, x2, y2 = map(int, bbox * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]))
            # Draw bounding box and label on the frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img

def main():
    st.title("Real-time Object Detection with YOLOv8")

    # Create an instance of YOLOv8 model
    model = YOLO('c:\\Users\\user\\Desktop\\cv\\epoch4.pt')

    # Display the webcam feed with object detection
    webrtc_streamer(key="example", video_transformer_factory=lambda: ObjectDetector(model))

if __name__ == "__main__":
    main()
