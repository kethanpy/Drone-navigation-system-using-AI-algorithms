import numpy as np
import cv2
import os
import json
import time
import requests
import io

from deepface import DeepFace


# ============================================================
# 1. CAMERA SETUP
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Web Camera open avvatledu")
    exit()

print("Camera started successfully...")


# ============================================================
# 2. TELEGRAM HTTP CONFIGURATION
# ============================================================

# IMPORTANT:
# Put your NEW Telegram BotFather token here
TELEGRAM_BOT_TOKEN = "8862936318:AAErty5xwRcIxeW2UIfk-UrNe_3-_6PxwS4"

# Your Telegram Chat ID
TELEGRAM_CHAT_ID = "6388181014"

chat_ids = [TELEGRAM_CHAT_ID]


# ============================================================
# 3. TELEGRAM HTTP FUNCTIONS
# ============================================================

def send_telegram_message(chat_id, message):

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    data = {
        "chat_id": chat_id,
        "text": message
    }

    try:

        response = requests.post(
            url,
            data=data,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        if result.get("ok"):
            print("Telegram message sent successfully")
        else:
            print("Telegram message failed:", result)

        return result

    except requests.exceptions.RequestException as e:

        print("Telegram message error:", e)

        return None


def send_telegram_photo(chat_id, image_bytes, caption=None):

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    files = {
        "photo": (
            "detected.jpg",
            image_bytes,
            "image/jpeg"
        )
    }

    data = {
        "chat_id": chat_id
    }

    if caption:
        data["caption"] = caption

    try:

        response = requests.post(
            url,
            data=data,
            files=files,
            timeout=20
        )

        response.raise_for_status()

        result = response.json()

        if result.get("ok"):
            print("Telegram photo sent successfully")
        else:
            print("Telegram photo failed:", result)

        return result

    except requests.exceptions.RequestException as e:

        print("Telegram photo error:", e)

        return None


# ============================================================
# 4. TEST TELEGRAM CONNECTION
# ============================================================

print("Testing Telegram connection...")

test_message = """
🚁 Drone Security System Started

Camera: ONLINE
YOLO: Loading
Face Recognition: Loading
Telegram HTTP API: Connected
"""

for cid in chat_ids:
    send_telegram_message(cid, test_message)


# ============================================================
# 5. LOAD PERSON DATABASE
# ============================================================

PERSON_DB_PATH = "persons/details.json"

if not os.path.exists(PERSON_DB_PATH):

    print("Person database not found:", PERSON_DB_PATH)
    exit()


with open(PERSON_DB_PATH, "r") as f:

    persons_data = json.load(f)


print("Person database loaded successfully")


# ============================================================
# 6. YOLO FILES
# ============================================================

YOLO_CONFIG = "yolov4-tiny.cfg"
YOLO_WEIGHTS = "yolov4-tiny.weights"
YOLO_CLASSES = "classes.txt"


# Check files

if not os.path.exists(YOLO_CONFIG):
    print("Missing:", YOLO_CONFIG)
    exit()

if not os.path.exists(YOLO_WEIGHTS):
    print("Missing:", YOLO_WEIGHTS)
    exit()

if not os.path.exists(YOLO_CLASSES):
    print("Missing:", YOLO_CLASSES)
    exit()


# ============================================================
# 7. LOAD YOLO CLASSES
# ============================================================

with open(YOLO_CLASSES, "r") as f:

    classes = [
        line.strip()
        for line in f.readlines()
    ]


print("YOLO classes loaded:", len(classes))


# ============================================================
# 8. LOAD YOLO MODEL
# ============================================================

net = cv2.dnn.readNetFromDarknet(
    YOLO_CONFIG,
    YOLO_WEIGHTS
)

layer_names = net.getLayerNames()

output_layers = [
    layer_names[i - 1]
    for i in net.getUnconnectedOutLayers()
]


print("YOLO model loaded successfully")


# ============================================================
# 9. SETTINGS
# ============================================================

PERSON_CLASS_ID = 0

CONFIDENCE_THRESHOLD = 0.5

NMS_THRESHOLD = 0.4

ALERT_COOLDOWN = 6

last_alert_time = 0


# ============================================================
# 10. FACE RECOGNITION SETTINGS
# ============================================================

FACE_MODEL = "VGG-Face"

DISTANCE_METRIC = "cosine"

FACE_THRESHOLD = 0.40


# ============================================================
# 11. MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Camera frame read failed")
        break


    # --------------------------------------------------------
    # YOLO OBJECT DETECTION
    # --------------------------------------------------------

    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (416, 416),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)

    outputs = net.forward(output_layers)


    boxes = []
    confidences = []
    class_ids = []


    # --------------------------------------------------------
    # PROCESS YOLO OUTPUT
    # --------------------------------------------------------

    for output in outputs:

        for detection in output:

            scores = detection[5:]

            class_id = np.argmax(scores)

            confidence = scores[class_id]


            if confidence > CONFIDENCE_THRESHOLD:

                center_x = int(
                    detection[0] * width
                )

                center_y = int(
                    detection[1] * height
                )

                w = int(
                    detection[2] * width
                )

                h = int(
                    detection[3] * height
                )


                x = int(center_x - w / 2)

                y = int(center_y - h / 2)


                boxes.append(
                    [x, y, w, h]
                )

                confidences.append(
                    float(confidence)
                )

                class_ids.append(
                    class_id
                )


    # --------------------------------------------------------
    # NON MAXIMUM SUPPRESSION
    # --------------------------------------------------------

    indexes = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD
    )


    detected_objects = []

    person_boxes = []


    if len(indexes) > 0:

        for i in indexes.flatten():

            x, y, w, h = boxes[i]

            class_id = class_ids[i]

            confidence = confidences[i]


            # Make coordinates safe

            x = max(0, x)

            y = max(0, y)

            w = min(w, width - x)

            h = min(h, height - y)


            if class_id < len(classes):

                class_name = classes[class_id]

            else:

                class_name = "unknown"


            detected_objects.append(
                f"{class_name} ({confidence:.2f})"
            )


            # ------------------------------------------------
            # PERSON DETECTED
            # ------------------------------------------------

            if class_id == PERSON_CLASS_ID:

                person_boxes.append(
                    (x, y, w, h)
                )


            # Draw object box

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )


            cv2.putText(
                frame,
                f"{class_name} {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )


    # ========================================================
    # 12. FACE RECOGNITION
    # ========================================================

    authorized = False

    person_name = "Unknown"

    person_id = ""

    person_branch = ""

    person_year = ""


    if len(person_boxes) > 0:

        # First detected person

        x, y, w, h = person_boxes[0]


        person_crop = frame[
            y:y + h,
            x:x + w
        ]


        if person_crop.size > 0:

            try:

                # ------------------------------------------------
                # TEMPORARY IMAGE FOR DEEPFACE
                # ------------------------------------------------

                temp_face_path = "temp_person.jpg"

                cv2.imwrite(
                    temp_face_path,
                    person_crop
                )


                # ------------------------------------------------
                # DEEPFACE FIND
                # ------------------------------------------------

                results = DeepFace.find(

                    img_path=temp_face_path,

                    db_path="persons",

                    model_name=FACE_MODEL,

                    distance_metric=DISTANCE_METRIC,

                    enforce_detection=False,

                    silent=True
                )


                # ------------------------------------------------
                # PROCESS RESULT
                # ------------------------------------------------

                if results:

                    if isinstance(results, list):

                        result_df = results[0]

                    else:

                        result_df = results


                    if not result_df.empty:

                        distance_column = None


                        # Find distance column

                        for column in result_df.columns:

                            if "cosine" in column.lower():

                                distance_column = column

                                break


                            if "distance" in column.lower():

                                distance_column = column

                                break


                        if distance_column:

                            distance = float(
                                result_df.iloc[0][
                                    distance_column
                                ]
                            )


                            if distance <= FACE_THRESHOLD:

                                authorized = True


                                identity_path = str(
                                    result_df.iloc[0][
                                        "identity"
                                    ]
                                )


                                # Get filename

                                filename = os.path.basename(
                                    identity_path
                                )


                                # Remove extension

                                person_key = os.path.splitext(
                                    filename
                                )[0]


                                # Search person information

                                if isinstance(
                                    persons_data,
                                    list
                                ):

                                    for person in persons_data:

                                        if (
                                            str(
                                                person.get("id")
                                            )
                                            == person_key
                                            or
                                            str(
                                                person.get("name")
                                            ).lower()
                                            ==
                                            person_key.lower()
                                        ):

                                            person_name = person.get(
                                                "name",
                                                "Unknown"
                                            )

                                            person_id = person.get(
                                                "id",
                                                ""
                                            )

                                            person_branch = person.get(
                                                "branch",
                                                ""
                                            )

                                            person_year = person.get(
                                                "year",
                                                ""
                                            )

                                            break


                                elif isinstance(
                                    persons_data,
                                    dict
                                ):

                                    person = persons_data.get(
                                        person_key
                                    )


                                    if person:

                                        person_name = person.get(
                                            "name",
                                            "Unknown"
                                        )

                                        person_id = person.get(
                                            "id",
                                            ""
                                        )

                                        person_branch = person.get(
                                            "branch",
                                            ""
                                        )

                                        person_year = person.get(
                                            "year",
                                            ""
                                        )


            except Exception as e:

                print(
                    "Face recognition error:",
                    e
                )


    # ========================================================
    # 13. DISPLAY AUTHORIZATION STATUS
    # ========================================================

    if len(person_boxes) > 0:

        x, y, w, h = person_boxes[0]


        if authorized:

            label = (
                f"AUTHORIZED: {person_name}"
            )


            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )


            cv2.putText(
                frame,
                label,
                (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        else:

            label = "UNAUTHORIZED PERSON"


            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                3
            )


            cv2.putText(
                frame,
                label,
                (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


    # ========================================================
    # 14. TELEGRAM ALERT
    # ========================================================

    current_time = time.time()


    if (
        len(person_boxes) > 0
        and
        current_time - last_alert_time
        >= ALERT_COOLDOWN
    ):

        # ----------------------------------------------------
        # Object information
        # ----------------------------------------------------

        if detected_objects:

            object_text = "\n".join(
                detected_objects
            )

        else:

            object_text = "No objects"


        # ----------------------------------------------------
        # AUTHORIZED MESSAGE
        # ----------------------------------------------------

        if authorized:

            alert_msg = f"""
🚁 DRONE SECURITY ALERT

✅ AUTHORIZED PERSON

👤 Name: {person_name}
🆔 ID: {person_id}
🏢 Branch: {person_branch}
📅 Year: {person_year}

📦 Detected Objects:
{object_text}

⏰ Time: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""


        # ----------------------------------------------------
        # UNAUTHORIZED MESSAGE
        # ----------------------------------------------------

        else:

            alert_msg = f"""
🚨 DRONE SECURITY ALERT

❌ UNAUTHORIZED PERSON DETECTED

👤 Name: Unknown

📦 Detected Objects:
{object_text}

⏰ Time: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""


        # ----------------------------------------------------
        # SEND TELEGRAM
        # ----------------------------------------------------

        success = False


        for cid in chat_ids:

            result = send_telegram_message(
                cid,
                alert_msg
            )


            if result and result.get("ok"):

                success = True


        # ----------------------------------------------------
        # SEND PHOTO
        # ----------------------------------------------------

        ret_img, img_buffer = cv2.imencode(
            ".jpg",
            frame
        )


        if ret_img:

            image_bytes = io.BytesIO(
                img_buffer.tobytes()
            )

            image_bytes.seek(0)


            for cid in chat_ids:

                send_telegram_photo(
                    cid,
                    image_bytes,
                    caption="🚁 Drone Security Camera"
                )


        # Update cooldown only after alert

        if success:

            last_alert_time = current_time

            print(
                "Telegram alert sent successfully"
            )


    # ========================================================
    # 15. SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Drone Security System",
        frame
    )


    # Press Q to quit

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# 16. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Drone Security System stopped.")