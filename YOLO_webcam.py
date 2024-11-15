import sys
from ultralytics import YOLO
import cv2
import math 
import torch
import os
from roboflow import Roboflow
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def get_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']

def load_model_from_roboflow(workspace, project_name, version_number, model_name,nb_epochs,batch_size) -> tuple[YOLO, any]:
    if not os.path.exists("data/" + model_name):
        rf = Roboflow(api_key="8DrZ8Cjqqu2mLaJM9iPH")
        project = rf.workspace(workspace).project(project_name)
        version = project.version(version_number)
        dataset = version.download("yolov8", "data/" + model_name)
    classNames = get_class_names("data/" + model_name + "/data.yaml")
    if os.path.exists("model/" + model_name + "_e" + str(nb_epochs) + "_b" + str(batch_size) + ".pt"):
        return YOLO("model/" + model_name + "_e" + str(nb_epochs) + "_b" + str(batch_size) + ".pt"), classNames
    return train_and_save_model(model_name,nb_epochs,batch_size), classNames

def train_and_save_model(model_name,nb_epochs=10,batch_size=8) -> YOLO:
    model = YOLO("yolo-Weights/yolov8n.pt")
    model.train(data="data/" + model_name + "/data.yaml", epochs=nb_epochs, batch=batch_size, device=device)
    model.save("model/" + model_name + "_e" + str(nb_epochs) + "_b" + str(batch_size) + ".pt")
    return model

def load_model(model_name,nb_epochs,batch_size) -> tuple[YOLO, any]:
    classNames = get_class_names("data/" + model_name + "/data.yaml")
    if os.path.exists("model/" + model_name + "_e" + str(nb_epochs) + "_b" + str(batch_size) + ".pt"):
        model = YOLO("model/" + model_name + "_e" + str(nb_epochs) + "_b" + str(batch_size) + ".pt")
    else:
        model = train_and_save_model(model_name,nb_epochs,batch_size)
    return model, classNames

def main():
    
    model_name = "rock-paper-scissors"
    print("Loading model " + model_name + "...")
    # model = load_model("hand-gesture")
    model, classNames = load_model_from_roboflow("roboflow-58fyf", "rock-paper-scissors-sxsw", 11, model_name, 20, 8)

    print("Classes --->", classNames)

    # start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    score_player1 = 0
    score_player2 = 0
    sign_player1 = ""
    sign_player2 = ""
    player1_first_detect = 100
    player2_first_detect = 100

    frame_count = 0

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Failed to capture image")
            continue

        # Mirror the image horizontally
        img = cv2.flip(img, 1)
        
        frame_count += 1

        with SuppressOutput():
            results = model(img, stream=True)

        # build ui with results
        cv2.rectangle(img, (0, 0), (637, 720), (0, 0, 255), 3)
        cv2.rectangle(img, (642, 0), (1280, 720), (255, 0, 0), 3)
        cv2.putText(img, "Player 1: " + str(score_player1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "Player 2: " + str(score_player2), (652, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, sign_player1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, sign_player2, (652, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # display progress bar
        cv2.rectangle(img, (0, 700), (round(frame_count / 50 * 1280), 720), (0, 255, 0), -1)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                x_mean = (x1 + x2) / 2

                if x_mean < 640:
                    if sign_player1 == "":
                        player1_first_detect = frame_count
                        sign_player1 = classNames[cls].lower()
                    elif frame_count - player1_first_detect < 10:
                        sign_player1 = classNames[cls].lower()
                    else:
                        continue
                else:
                    if sign_player2 == "":
                        player2_first_detect = frame_count
                        sign_player2 = classNames[cls].lower()
                    elif frame_count - player2_first_detect < 10:
                        sign_player2 = classNames[cls].lower()
                    else:
                        continue

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        if(frame_count >= 50):
            if sign_player1 == "rock" and sign_player2 == "scissors":
                score_player1 += 1
            elif sign_player1 == "scissors" and sign_player2 == "rock":
                score_player2 += 1
            elif sign_player1 == "scissors" and sign_player2 == "paper":
                score_player1 += 1
            elif sign_player1 == "paper" and sign_player2 == "scissors":
                score_player2 += 1
            elif sign_player1 == "rock" and sign_player2 == "paper":
                score_player2 += 1
            elif sign_player1 == "paper" and sign_player2 == "rock":
                score_player1 += 1
            sign_player1 = ""
            sign_player2 = ""
            player1_first_detect = 100
            player2_first_detect = 100
            frame_count = 0

        cv2.imshow('Rock Paper Scissors Game', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()