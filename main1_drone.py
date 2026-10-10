import cv2 as cv
import time
import telepot
import numpy as np
import io


import numpy as np
import requests
import urllib
import time

# ESP32-CAM IP address
esp32cam_url = 'http://192.168.199.53/640x480.jpg'

time.sleep(2)
# Function to fetch images from ESP32-CAM
def get_esp32cam_image():
    try:
        response = requests.get(esp32cam_url, timeout=10)
        if response.status_code == 200:
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            img = cv.imdecode(img_array, -1)
            return img
    except Exception as e:
        print(f"Error fetching image from ESP32-CAM: {str(e)}")
    return None


# Define thresholds and colors
Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Load class names from the text file
class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Load YOLO model
net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Initialize Telegram Bot
bot = telepot.Bot('7088533246:AAHA0FxEFn1xjVR_S2HF7uT9V1ahOqVkjbk')  # Replace with your Telegram bot API key
chat_id = '1585906163'  # Replace with your chat ID

# Open video capture
cap = cv.VideoCapture(0)
starting_time = time.time()
frame_counter = 0

# Define detection class IDs for 'person', 'car', and 'bike'
BIKE_CLASS_ID = 0
PERSON_CLASS_ID = 1
CAR_CLASS_ID = 2
MBIKE_CLASS_ID = 3
while True:
    #ret, frame = cap.read()
    frame_counter += 1
    frame = get_esp32cam_image()
    #if not ret:
     #   break

    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %.2f" % (class_name[classid], score)
        cv.rectangle(frame, box, color, 1)
        cv.putText(frame, label, (box[0], box[1]-10),
                   cv.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
        
        # Check if the detected object is a person, car, or bike
        if classid == PERSON_CLASS_ID or classid == CAR_CLASS_ID or classid == BIKE_CLASS_ID or classid == MBIKE_CLASS_ID:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            # Send message to Telegram bot
            message = f"Detected {class_name[classid]} at {current_time}"
            bot.sendMessage(chat_id, message)
            
            # Convert frame to a byte array and send it as a photo
            ret, img_buffer = cv.imencode('.jpg', frame)
            if ret:
                # Use BytesIO to simulate a file-like object in memory
                img_bytes = io.BytesIO(img_buffer)
                img_bytes.seek(0)  # Reset pointer to the start of the byte stream
                bot.sendPhoto(chat_id, photo=img_bytes)
    
    # Calculate FPS
    endingTime = time.time() - starting_time
    fps = frame_counter / endingTime
    cv.putText(frame, f'FPS: {fps:.2f}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv.imshow('frame', frame)
    
    # Break if 'q' is pressed
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# Clean up
cap.release()
cv.destroyAllWindows()
