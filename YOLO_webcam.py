from ultralytics import YOLO
import cv2
import math 
import torch
import os
from roboflow import Roboflow
import yaml

def get_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']

def load_rock_paper_scissors_model():
    return load_model_from_roboflow("roboflow-58fyf", "rock-paper-scissors-sxsw", 11, "rock-paper-scissors")

def load_model_from_roboflow(workspace, project_name, version_number, model_name):
    if os.path.exists("model/" + model_name + ".pt"):
        return YOLO("model/" + model_name + ".pt")
    if not os.path.exists("data/" + model_name):
        rf = Roboflow(api_key="8DrZ8Cjqqu2mLaJM9iPH")
        project = rf.workspace(workspace).project(project_name)
        version = project.version(version_number)
        dataset = version.download("yolov8", "data/" + model_name)
    return train_and_save_model(model_name)

def train_and_save_model(model_name):
    model = YOLO("yolo-Weights/yolov8n.pt")
    model.train(data="data/" + model_name + "/data.yaml", epochs=10, batch=8, device="cuda")
    model.save("model/" + model_name + ".pt")
    return model

def load_model(model_name):
    if os.path.exists("model/" + model_name + ".pt"):
        model = YOLO("model/" + model_name + ".pt")
    else:
        model = train_and_save_model(model_name)
    return model

def main():
    # model
    # model = load_model("hand-gesture")
    model = load_rock_paper_scissors_model()

    # object classes
    # classNames = ['Down', 'Left', 'Right', 'Stop', 'Thumbs Down', 'Thumbs up', 'Up']
    # classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    #               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    #               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    #               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    #               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    #               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    #               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    #               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    #               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    #               "teddy bear", "hair drier", "toothbrush"
    #               ]
    classNames = get_class_names("data/rock-paper-scissors/data.yaml")
    # classNames = get_class_names("data/rock-paper-scissors/data.yaml")
    print("Classes --->",classNames.__len__())

    # start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            continue

        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()