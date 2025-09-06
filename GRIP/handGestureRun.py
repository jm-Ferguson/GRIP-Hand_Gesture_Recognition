import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn

gestureStr = "" # String to keep track of current gesture
Color = (0,0,0) # Color of text on video


class HandModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=42, out_features=256)
        self.layer_2 = nn.Linear(in_features=256, out_features=256)
        self.layer_3 = nn.Linear(in_features=256, out_features=256)
        self.layer_4 = nn.Linear(in_features=256, out_features=128)
        self.layer_5 = nn.Linear(in_features=128, out_features = 5)
        self.layer_6 = nn.ReLU()

    def forward(self, x):
        return self.layer_5(self.layer_6(self.layer_4(self.layer_6(self.layer_3(self.layer_6(self.layer_2(self.layer_6(self.layer_1(x)))))))))

loaded_model_handGes = HandModelV1()


loaded_model_handGes.load_state_dict(torch.load(f="models/03_HandGesture_Binary.pth"))

X_string = ""


# Normalize Hand Data Method
# - Normalize all of landmark data to be origined at bottom of palm ID(0)
def normalize_data(dataString):
    lines = dataString.strip().split("\n")
    normOutput = []
    handData = []

    for line in lines:
        parts = line.split()  # Split by spaces
        if len(parts) == 3:
            id_ = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            handData.append([id_, x, y])
        
        if len(handData) == 21:  # 21 landmarks per hand
            base_x, base_y = handData[0][1], handData[0][2]
            for id_, x, y in handData:
                norm_x = x - base_x
                norm_y = y - base_y
                normOutput.append(f"{id_} {norm_x} {norm_y}")
            handData = []

    return "\n".join(normOutput)
#

#
#
# Convert string into array with correct shape
def convert_toArray(dataString):

    # Split and setup
    lines = dataString.strip().split("\n")
    landmarks = []
    #
    
    # Parse each line, extracting the x and y coordinates
    for line in lines:
        parts = line.split()
        if len(parts) == 3: 
            x = float(parts[1])
            y = float(parts[2])
            landmarks.append([x, y])
    
    # Convert the landmarks list into a NumPy array
    landmarks_array = np.array(landmarks)
    
    # Fail-safe
    total_landmarks = len(landmarks_array)
    if total_landmarks % 21 != 0:
        print(f"Warning: ({total_landmarks}) is not divisible by 21.")
        landmarks_array = landmarks_array[:-(total_landmarks % 21)]
    
    # Reshape (number of samples, 21 landmarks, 2 coordinates)
    reshaped_array = landmarks_array.reshape(-1, 21, 2)
    
    # Flatten (21 landmarks, 2 coordinates) -> 1D array (42 features)
    flattened_array = reshaped_array.reshape(-1, 42)
    
    return flattened_array
#

###########################################################################
vid = cv2.VideoCapture(0)
vid.set(3, 1280)
mphands = mp.solutions.hands
Hands = mphands.Hands(max_num_hands= 1, min_detection_confidence= 0.8, min_tracking_confidence= 0.65 )
mpdraw = mp.solutions.drawing_utils
while True :
    _, frame = vid.read()
    RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# convert from bgr to rgb
    result = Hands.process(RGBframe)

    # If a hand is found / landmarks are created
    if result.multi_hand_landmarks:

        # For-each landmark
        for handLm in result.multi_hand_landmarks :
            # Drawing each landmark
            mpdraw.draw_landmarks(frame, handLm, mphands.HAND_CONNECTIONS,
                                  mpdraw.DrawingSpec(color=(178, 135, 255), circle_radius=7, # CIRCLES
                                                     thickness=cv2.FILLED),
                                  mpdraw.DrawingSpec(color=(240, 221, 63), thickness=7) # LINES 
                                  )
            # Looks at each landmark individually / creating string data for recognition AI
            for id, lm in enumerate(handLm.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id < 20:
                    X_string += (f"{id} {lm.x} {lm.y}\n")
                else:
                    X_string += (f"{id} {lm.x} {lm.y}")
                
                
    # If data string isnt empty
    if X_string != "":
        # Setup data
        normalized_string = normalize_data(X_string)
        reshaped_array = convert_toArray(normalized_string)
        X = reshaped_array
        X = torch.from_numpy(X).type(torch.float)
        #

        loaded_model_handGes.eval() # Eval mode for AI

        with torch.inference_mode():
            loaded_model_preds = loaded_model_handGes(X)
            test_pred = torch.argmax(torch.softmax(loaded_model_preds, dim=1), dim=1)

        # Get what specific gesture the recognition AI predicts it to be
        gesture = int(test_pred.item())
        match gesture:
            case 0:

                gestureStr = "No Gesture"
                color = (0,255,0)
            case 1:

                gestureStr = "I Love You"
                color = (0,255,0)
            case 2:

                gestureStr = "Peace / Two"
                color = (0,255,0)
            case 3:

                gestureStr = "Three"
                color = (0,255,0)
            case 4:

                gestureStr = "Rad"
                color = (0,255,0)
            case _:
                gestureStr = ""
                color = (0,0,0)
        #####

        X_string = "" # resets data
    else:
        gestureStr = "No Hand"
        color = (0,0,255)

    cv2.putText(frame, gestureStr, (15,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("video", frame)
    cv2.waitKey(1)