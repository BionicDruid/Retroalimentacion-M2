import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def normalization(data):
    #This normalization utilizes Z-score
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data-mean)/std
    

def sigmoid(z):
    # Handle large positive and negative values of z
    # Use np.clip to avoid overflow in the exponential function
    z = np.clip(z, -700, 700)  # Clipping values to avoid overflow
    return 1 / (1 + np.exp(-z))

def r_squared(y_true, y_pred):
    mean_y = np.mean(y_true)
    sumsquares_total = np.sum((y_true - mean_y) ** 2)
    sumsquares_prediction = np.sum((y_true - y_pred) ** 2)
    return 1 - (sumsquares_prediction / sumsquares_total) if sumsquares_total != 0 else 0


def precision(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def confusion_matrix(y_true, y_pred, threshold=0.5):
    TP = 0 #true positive
    TN = 0 #true negative
    FP = 0 #false positive
    FN = 0 #false negative
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    predicted = (y_pred >= threshold).astype(int)
    #Check if the prediction is correct
    TP = np.sum((predicted == 1) & (y_true == 1))
    TN = np.sum((predicted == 0) & (y_true == 0))
    FP = np.sum((predicted == 1) & (y_true == 0))
    FN = np.sum((predicted == 0) & (y_true == 1))

    accuracy = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) > 0 else 0 #avoid dividing by zero and crashing the program
    specifity = (TN)/(TN+FP) if (TN+FP) > 0 else 0 #avoid dividing by zero and crashing the program
    precision = (TP)/(TP+FP) if (TP+FP) > 0 else 0 #avoid dividing by zero and crashing the program
    recall = (TP)/(TP+FN) if (TP+FN) > 0 else 0 #avoid dividing by zero and crashing the program
    F1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0 #avoid dividing by zero and crashing the program
    
    return {'True Postives: ': TP, 'True Negatives: ': TN,
             'False Postives:': FP, 'False Negatives: ': FN,
               'Accuracy': accuracy, 'Specifity': specifity,
                 'Precision': precision, 'F1 Score': F1,
                   'Recall: ': recall}


def linear(params, sample):
    return np.dot(params, sample)   # Apply dot function to the parameters and the sample

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Prevent log(0) issues
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def Gradient_Descent(params, samples, y, alfa):
    adjusted = np.copy(params) # Doesn't modify the original array
    for i in range (len(params)):
        c = 0
        for j in range(len(samples)):
            prediction = sigmoid(linear(params, samples[j]))
            error = prediction - y[j] #Compute the prediction error
            c += error*samples[j][i] #C gets updated based on the error and the sample
        adjusted[i] = params[i]-alfa*(1/len(samples))*c #Update the parameter
    return adjusted


def logistic_regression(x_train, y_train, x_test, y_test, alfa):
   # Normalizing the data
    normalized_x_train = normalization(x_train)
    normalized_x_test = normalization(x_test)

    # Initial parameters
    params = np.zeros(x_train.shape[1]) #Set the parameters to an array of zeros 
    previous_error = float('inf') #Set the previous error to infinity
    epochs = 0

    while True:
        # Update the parameters using gradient descent
        params = Gradient_Descent(params, normalized_x_train, y_train, alfa)
        
        # Predict the values for the test set
        y_pred_prob = [sigmoid(linear(params, sample)) for sample in normalized_x_test]
        y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred_prob]
        
        

        #Calculate the error
        current_error = binary_cross_entropy(y_test, y_pred_prob)
        
        # Print results for the current epoch
        epochs += 1
        print(f'Epoch: {epochs}')
        print('Parameters:', params)
        print('Current error:', current_error)
        print('Previous error:', previous_error)
        print('--------------------------------------')
        print('R^2 test and prediction:', r_squared(y_test, y_pred))
        print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
        print('......................................')

        if abs(previous_error - current_error) < 0.000001:
            print("Finished training")
            break
        previous_error = current_error

    return params
    


#To improve the model we'll change metascore to binary labels
def metascore_to_binary(metascore):
    return 1 if metascore > 50 else 0
    
# Load the data
training_data = pd.read_csv('training_dataset.csv', header=0)
validation_data = pd.read_csv('validation_dataset.csv', header=0)

# Convert Metascore to binary labels
training_data['Binary_Label'] = training_data['Metascore'].apply(metascore_to_binary)
validation_data['Binary_Label'] = validation_data['Metascore'].apply(metascore_to_binary)

# Extract features and labels
x_train = training_data[['Rating', 'Votes', 'Revenue (Millions)']].values.astype(float)
y_train = training_data['Binary_Label'].values.astype(int)

x_test = validation_data[['Rating', 'Votes', 'Revenue (Millions)']].values.astype(float)
y_test = validation_data['Binary_Label'].values.astype(int)

# Run logistic regression
logistic_regression(x_train, y_train, x_test, y_test, 0.01)