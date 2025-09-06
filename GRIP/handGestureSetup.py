import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from pathlib import Path


# -- Raw Data String Section
with open("rawData.txt", "r", encoding="utf-8") as file:
    dataString = file.read()
##################################################################################################################################-End Section

# -- Data Setup Section:

# normalize_data Method
# - Normalize all of landmark data to be origined at bottom of palm ID(0)
def normalize_data(dataString):

    # Stripping and setting up
    lines = dataString.strip().split("\n")
    normOutput = []
    handData = []
    #

    # Loop through each line and evaluate (x, y) for each one
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
normalized_string = normalize_data(dataString) # Setup string for new normalized data


# Conv Method
# - Convert string into array with the correct shape
def convert_toArray(dataString):
    
    # Split and setup
    lines = dataString.strip().split("\n")
    landmarks = []
    #
    
    # Loop through each line creating a list
    for line in lines:
        parts = line.split()
        if len(parts) == 3:
            x = float(parts[1])
            y = float(parts[2])
            landmarks.append([x, y])
    
    
    landmarks_array = np.array(landmarks) # Convert the landmarks list into a NumPy array
    
    # Fail-safe and organiziation
    total_landmarks = len(landmarks_array)
    if total_landmarks % 21 != 0:
        print(f"Warning: The total number of landmarks ({total_landmarks}) is not divisible by 21. Trimming the extra data.")
        # Trim extra rows to ensure full sets of 21 landmarks
        landmarks_array = landmarks_array[:-(total_landmarks % 21)]
    
    # Reshape (number of samples, 21 landmarks, 2 coordinates)
    reshaped_array = landmarks_array.reshape(-1, 21, 2)
    
    # Flatten (21 landmarks, 2 coordinates) -> 1D array (42 features)
    flattened_array = reshaped_array.reshape(-1, 42)
    
    return flattened_array
#
reshaped_array = convert_toArray(normalized_string) # reshapedArr variable of data


# Create arr, size 640; first 320 elements as 0, the next 80 as 1, next 80 as 2, next 80 as 3, next 80 as 4
# 'Label' Array
labels_array = np.concatenate((np.zeros(320), 
                               np.ones(80),
                               np.full(80,2),
                               np.full(80,3),
                               np.full(80,4)
                               ))

##################################################################################################################################-End Section


# Hand Gesture AI Setup Section:

X = reshaped_array # Inputs / array data
X = torch.from_numpy(X).type(torch.float) #conv to torch float array
y = labels_array
y = torch.from_numpy(y).type(torch.long) #conv to torch long arr

# Create a training and testing split on the data (80% training and 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=41)

device = "cuda" if torch.cuda.is_available() else "cpu" # Setting up gpu acceleration if possible

# Hand Gesture Model
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
#   
model_1 = HandModelV1().to(device)


loss_fn = nn.CrossEntropyLoss() # Loss Function


optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)#Changed from SGD

# Calculation of accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc


# Training LOOP
torch.manual_seed(41)
torch.cuda.manual_seed(41)
epochs = 250

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device),y_test.to(device)

for epoch in range(epochs):
    ##Training
    model_1.train()

    #Forward Pass
    y_logits = model_1(X_train)
    y_pred = torch.argmax(y_logits, dim=1)

    # Loss / acc
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    #Optim
    optimizer.zero_grad()

    # Loss back
    loss.backward()

    # Step optim
    optimizer.step()

    
    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test)
        test_pred = torch.argmax(torch.softmax(test_logits, dim=1), dim=1)

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    if epoch % 50 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")
##################################################################################################################################-End Section


### Save Model Section:
print("\n\nWOULD YOU LIKE TO SAVE?")
userInput = input()

if userInput == "yes":

    print("\nNAME OF MODEL (NO SPACES):")
    name_of_model = input()

    MODEL_PATH = Path("modelsTest")
    MODEL_PATH.mkdir(parents=True)

    MODEL_NAME = name_of_model + ".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)
#########################-End Section

