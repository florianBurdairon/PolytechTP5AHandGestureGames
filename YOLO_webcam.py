# Import libraries 
from ultralytics import YOLO, settings
import cv2
import math 
import torch
import os
from roboflow import Roboflow
import yaml

# Select the device to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Select the directory where the datasets are stored
settings.update(datasets_dir='.')

def get_class_names(yaml_path):
    """Get class names from yaml file"""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']

def load_model_from_roboflow(workspace, project_name, version_number, model_name,nb_epochs,batch_size) -> tuple[YOLO, any]:
    """
    Load model from Roboflow

    If the dataset is not in `data/` directory, then it will be downloaded

    If the model is not in `model/` directory, then it will be trained and its weights will be saved
    """
    # Download dataset if not exists
    if not os.path.exists("data/" + model_name):
        rf = Roboflow(api_key="8DrZ8Cjqqu2mLaJM9iPH")
        project = rf.workspace(workspace).project(project_name)
        version = project.version(version_number)
        dataset = version.download("yolov8", "data/" + model_name)
    
    # Get class names from yaml file
    classNames = get_class_names("data/" + model_name + "/data.yaml")

    # Load model if exists
    if os.path.exists("model/" + model_name + "_e" + str(nb_epochs) + "_b" + str(batch_size) + ".pt"):
        return YOLO("model/" + model_name + "_e" + str(nb_epochs) + "_b" + str(batch_size) + ".pt"), classNames
    
    # If model does not exist, then train and save it
    return train_and_save_model(model_name,nb_epochs,batch_size), classNames

def train_and_save_model(model_name,nb_epochs=10,batch_size=8) -> YOLO:
    """
    Train and save model

    Model will be saved in `model/` directory
    """
    # Initialize model with pre-trained weights
    model = YOLO("yolo-Weights/yolov8n.pt")

    # Train model on custom dataset
    model.train(data="data/" + model_name + "/data.yaml", epochs=nb_epochs, batch=batch_size, device=device)

    # Save model in `model/` directory
    model.save("model/" + model_name + "_e" + str(nb_epochs) + "_b" + str(batch_size) + ".pt")

    return model

def load_model(model_name,nb_epochs,batch_size) -> tuple[YOLO, any]:
    """
    Load model
    
    The dataset should be in `data/` directory

    If the model is not in `model/` directory, then it will be trained and its weights will be saved
    """
    # Get class names from yaml file
    classNames = get_class_names("data/" + model_name + "/data.yaml")

    # Load model if exists
    if os.path.exists("model/" + model_name + "_e" + str(nb_epochs) + "_b" + str(batch_size) + ".pt"):
        model = YOLO("model/" + model_name + "_e" + str(nb_epochs) + "_b" + str(batch_size) + ".pt")
    # If model does not exist, then train and save it
    else:
        model = train_and_save_model(model_name,nb_epochs,batch_size)

    return model, classNames

def main():
    
    # Loading model
    model_name = "rock-paper-scissors"
    print("Loading model " + model_name + "...")
    model, classNames = load_model_from_roboflow("roboflow-58fyf", "rock-paper-scissors-sxsw", 11, model_name, 50, 8)

    print("Classes --->", classNames)

    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Initialize game variables
    score_player1 = 0
    score_player2 = 0
    sign_player1 = ""
    sign_player2 = ""
    player1_first_detect = 100
    player2_first_detect = 100

    frame_count = 0

    # Start game loop
    while cap.isOpened():
        # Capture a frame
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            continue

        # Mirror the image horizontally
        img = cv2.flip(img, 1)
        
        frame_count += 1

        # Detect objects in the image using YOLO
        results = model(img, stream=True)

        # Build UI with scores and signs
        cv2.rectangle(img, (0, 0), (637, 720), (0, 0, 255), 3)
        cv2.rectangle(img, (642, 0), (1280, 720), (255, 0, 0), 3)
        cv2.putText(img, "Player 1: " + str(score_player1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "Player 2: " + str(score_player2), (652, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, sign_player1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, sign_player2, (652, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Display progress bar for 50 frames to show the time limit for the round
        cv2.rectangle(img, (0, 700), (round(frame_count / 50 * 1280), 720), (0, 255, 0), -1)

        # Loop through the results and draw bounding boxes
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # Get confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # Get class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # Get the mean of x coordinates
                x_mean = (x1 + x2) / 2

                # Check if the object is on the left or right side of the screen
                if x_mean < 640:
                    # Player 1
                    # Check if the sign is detected for the first time or not
                    if sign_player1 == "":
                        player1_first_detect = frame_count
                        sign_player1 = classNames[cls].lower()
                    # Check if the sign is detected again within 20 frames
                    elif frame_count - player1_first_detect < 20:
                        sign_player1 = classNames[cls].lower()
                    else:
                        continue
                else:
                    # Player 2
                    # Check if the sign is detected for the first time or not
                    if sign_player2 == "":
                        player2_first_detect = frame_count
                        sign_player2 = classNames[cls].lower()
                    # Check if the sign is detected again within 20 frames
                    elif frame_count - player2_first_detect < 20:
                        sign_player2 = classNames[cls].lower()
                    else:
                        continue

                # Put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Put object details in cam
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        # Check if the round is over
        if(frame_count >= 50):
            # Check who wins the round and update the scores
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

            # Reset the signs and frame count for the next round
            sign_player1 = ""
            sign_player2 = ""
            player1_first_detect = 100
            player2_first_detect = 100
            frame_count = 0

        # Display the image
        cv2.imshow('Rock Paper Scissors Game', img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Clear cuda cache before running the code
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()