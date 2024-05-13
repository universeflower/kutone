# Import required libraries
import cv2
import streamlit as st
from ultralytics import YOLO

# Replace the relative path to your weight file
model = YOLO("best.pt")

# Setting page layout
st.set_page_config(
    page_title="Object Detection",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
)

# Creating main page heading
st.title("Object Detection")

# Adding image to the first column if image is uploaded
st.header("WebCam")
# 이미지 컨테이너 생성
web_cam_container = st.empty()
containers = {
    "can": None,
    "vinyl": None,
    "can_damaged": None,
    "pet": None,
    "paper": None,
    "plastic": None,
    "pet_damaged": None
}

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        cv2_img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        # Update the image container
        web_cam_container.image(cv2_img_rgb, caption="WebCam", use_column_width=True)

        for result in results:
            clist = result.boxes.cls
            cls = []
            for cno in clist:
                cls.append(model.names[int(cno)])
            
            boxes = result.boxes.xyxy.tolist()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cropped_object = cv2_img_rgb[int(y1):int(y2), int(x1):int(x2)]
                
                # Ensure that the index i is within the range of cls
                if i < len(cls):
                    label = cls[i]
                    
                    # Check if container exists, if not, create it
                    if containers[label] is None:
                        containers[label] = st.sidebar.empty()
                    
                    container = containers[label]
                    if label == 'pet':
                        st.toast("!!!찌그러뜨려주세요!!!")
                        st.toast("!!!라벨을 떼주세요!!!")
                    # Update container image
                    container.image(cropped_object, caption=f"{label.capitalize()} {i}", use_column_width=True)
                    
                        
                else:
                    # Handle the case where the number of labels doesn't match the number of boxes
                    # You can log an error or handle it as you see fit
                    pass

        # Check if any containers need to be removed
        for label, container in containers.items():
            if container is not None:
                if label not in cls:
                    container.empty()
                    containers[label] = None

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
