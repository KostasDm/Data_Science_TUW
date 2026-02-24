import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pennylane as qml
from pennylane import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import math 



#global flag for data reuploading
number_qubits = 2

# dataset : [number_layers, batch_size, optimizer]
initializer = {
    
'gluehweindorf' : [1,
                   1e6, # very large number to satisfy condition later 
                   qml.GradientDescentOptimizer(0.05)],

'lebkuchenstadt' : [3,
                    32, 
                    qml.RMSPropOptimizer(0.02)],

'krampuskogel' :  [5,
                   28, 
                   qml.AdamOptimizer(0.01)]

}



# Quantum device with 2 qubits
dev = qml.device("default.qubit", wires=number_qubits)


def data_preprocessing(df):

  target_class = "label"

  X = df.drop(columns = target_class)
  y = df[target_class]
  # shift label from {0, 1} to {-1, 1}
  y = 2*y - 1

  # range in [-pi, pi], also because we use angle encoding later
  scaler = MinMaxScaler(feature_range=(-math.pi, math.pi))

  scaler.fit(X)
  X_scaled = scaler.transform(X)

  return X_scaled, y


def train_test_split(X, y, test_split=0.25, random_state=42):


    np.random.seed(random_state)

    if len(X)==len(y):
       class_length = len(y)
       # shuffle indices randomly   
       indices = np.random.permutation(class_length)
      
       test_size = int(class_length * test_split)

       #get 25% of indices for test data
       test_idx = indices[:test_size]
       #get 75% of indices for train data
       train_idx = indices[test_size:]

       X_train, X_test = X[train_idx], X[test_idx]
       y_train, y_test = y[train_idx], y[test_idx]
     
       return X_train, y_train, X_test, y_test

    else:
        print("Length of features and target class do not match. Please debugg!")

        return 

    
def angle_encoding(X):

     qml.AngleEmbedding(features=X, wires=range(number_qubits), rotation="Y")
     

# following the format of pennylane tutorial
@qml.qnode(dev)
def variational_classifier(weights, X, data_reupload = False):
    

    # take number of layers from how many weight rows we have
    layers = weights.shape[0]

    for layer in range(layers):

      if not data_reupload:
        # then execute it only once
        if layer == 0: 
            angle_encoding(X)
      else:
        # do encoding for each layer -- data re-uploading
        angle_encoding(X)

      for wire in range(len(X)):
        qml.RY(weights[layer, wire], wires=wire)

      # we have only two qubits, thus it is only [0,1] for the given datasets   
      qml.CNOT(wires=[0, 1])
      #qml.CNOT(wires=[1, 0]) # tried it experimentally but did not see any improvement

    # return expectation value 
    return qml.expval(qml.PauliZ(0))
    

def prediction(weights, bias, X, data_reupload=False):

    # add classical bias to circuit
    predictions = [variational_classifier(weights, x, data_reupload) + bias for x in X]

    return predictions


def cost(weights, bias, X, y, data_reupload=False):

    # add classical bias to circuit
    predictions = [variational_classifier(weights, x, data_reupload) + bias for x in X]

    return loss_function(y, predictions)


def loss_function(y, predictions):

    return np.mean((y - qml.math.stack(predictions)) ** 2)


def accuracy(y, predictions):

    predictions = np.sign(predictions)
    acc = sum(abs(l - p) < 1e-6 for l, p in zip(y, predictions))
    return acc / len(y)




def train_model(weights_init, bias_init, batch_size, opt, X_train, y_train, X_test, y_test, data_reupload=False, steps=100):

    # superfluous
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    best_loss = 1e9 # arbitrary very large number
    tol = 1e-5 # tolerance
    stopping_criterion = 10

    weights = weights_init
    bias = bias_init

    metrics_dict = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": []
    }
    
    # only first two arguments, namely weights and bias
    grad_obj = qml.grad(cost, argnum=[0, 1])

    for s in range(steps):

        batch_index = np.random.randint(0, len(X_train), size=batch_size)
    
        X_batch = X_train[batch_index]
      
        Y_batch = y_train[batch_index].astype(float)

        # calculate gradient norm of given weight, bias
        grad_weight, grad_bias = grad_obj(weights, bias, X_batch, Y_batch, data_reupload)

        grad_norm = np.sqrt(np.sum(grad_weight**2) + np.sum(grad_bias**2))

        # update weight and bias
        weights, bias = opt.step(cost, weights, bias, X=X_batch, y=Y_batch,  data_reupload=data_reupload)


        # make prediction based on new weights and bias
        predictions_train = prediction(weights, bias, X_train, data_reupload)
        predictions_test = prediction(weights, bias, X_test, data_reupload)

        current_train_loss = loss_function(y_train, predictions_train)
        current_test_loss = loss_function(y_test, predictions_test)

        current_train_acc = accuracy(y_train, predictions_train)
        current_test_acc = accuracy(y_test, predictions_test)


        print(f"Epoch: {s+1:4d}\n" 
              f"Train-set loss: {current_train_loss:3f}\n" 
              f"Test-set loss: {current_test_loss:3f}\n"
              f"Train-set accuracy: {current_train_acc:.3f}\n"
              f"Test-set accuracy: {current_test_acc:.3f}\n"
              f"Grad-norm: {grad_norm:.3f}")

        
        metrics_dict["train_loss"].append(current_train_loss)
        metrics_dict["test_loss"].append(current_test_loss)
        metrics_dict["train_acc"].append(current_train_acc)
        metrics_dict["test_acc"].append(current_test_acc)



        if current_train_loss < best_loss - tol:

            best_loss = current_train_loss
            counter = 0
        else:
            counter +=1

        if counter >= stopping_criterion:

            print(f"Stopping criterion reached, after {counter} iterations, the loss has not improved any more") 
            break


    

    return weights, bias, metrics_dict


def run_model(dataset, X_train, y_train, X_test, y_test, data_reupload):

    layers = initializer[dataset][0]

    # Re-initialize weights & bias based on layers and input features
    weights_init = 0.1 * np.random.randn(layers, X_train.shape[1], requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)

    # Batch size
    batch_size = min(initializer[dataset][1], len(X_train))

    # Re-initialize optimizer and get step size

    old_opt = initializer[dataset][2]
    if hasattr(old_opt, "stepsize"):  
        stepsize = old_opt.stepsize
    
    else:
        stepsize = 0.01  # default 

    opt = type(old_opt)(stepsize)

    # Train the model
    weights, bias, metrics = train_model(weights_init, bias_init, batch_size, opt, X_train, y_train, X_test, y_test, data_reupload, steps=100)

    return weights, bias, metrics


def plot_confusion_matrix(y_test, predictions, title):


    cm = confusion_matrix(y_test, predictions, labels=[-1, 1])
    plt.figure(figsize=(7, 5)) 
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.show()


def plot_accuracy(metrics_dict, metrics_dict_reupload, title):
    plt.figure(figsize=(7,5))
    plt.plot(metrics_dict["train_acc"], label="Train (no re-upload)")
    plt.plot(metrics_dict["test_acc"], label="Test (no re-upload)")
    plt.plot(metrics_dict_reupload["train_acc"], "--", label="Train (re-upload)")
    plt.plot(metrics_dict_reupload["test_acc"], "--", label="Test (re-upload)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def plot_loss(metrics_dict, metrics_dict_reupload, title):
    plt.figure(figsize=(7,5))
    plt.plot(metrics_dict["train_loss"], label="Train (no re-upload)")
    plt.plot(metrics_dict["test_loss"], label="Test (no re-upload)")
    plt.plot(metrics_dict_reupload["train_loss"], "--", label="Train (re-upload)")
    plt.plot(metrics_dict_reupload["test_loss"], "--", label="Test (re-upload)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()




def run_exercise1():

    '''  Part 1-4 for all 3 datasets '''

    # Loading datasets as dataframes

    df_glueh = pd.read_csv("gluehweindorf.csv")
    df_kramp = pd.read_csv("krampuskogel.csv")
    df_leb = pd.read_csv("lebkuchenstadt.csv")

   
    ''' Gluehweindort dataset  '''

    #### Preprocessing
    X_glueh, y_glueh = data_preprocessing(df_glueh)


    ### Split data into train-test
    X_train_glueh, y_train_glueh, X_test_glueh, y_test_glueh = train_test_split(X_glueh, y_glueh)

    print('Dataset Gluehweindorf without data reuploading')

    weights_glueh, bias_glueh, metrics_glueh = run_model('gluehweindorf', X_train_glueh, y_train_glueh, X_test_glueh, y_test_glueh, False) 
   
    y_pred_glueh = np.sign(prediction(weights_glueh, bias_glueh, X_test_glueh, False))
    
    '''  Part 5 '''
    print('Dataset Gluehweindorf with data reuploading')
    weights_glueh_re, bias_glueh_re, metrics_glueh_re = run_model('gluehweindorf', X_train_glueh, y_train_glueh, X_test_glueh, y_test_glueh, True)

    y_pred_glueh_re = np.sign(prediction(weights_glueh_re, bias_glueh_re, X_test_glueh, True))


    
    ''' Lebkuchenstadt dataset  '''

    #### Preprocessing
    X_leb, y_leb = data_preprocessing(df_leb)
     

    ### Split data into train-test
    X_train_leb, y_train_leb, X_test_leb, y_test_leb = train_test_split(X_leb, y_leb)
 
    print('Dataset Lebkuchenstadt without data reuploading')
    weights_leb, bias_leb, metrics_leb = run_model('lebkuchenstadt',X_train_leb, y_train_leb, X_test_leb, y_test_leb, False)

    y_pred_leb = np.sign(prediction(weights_leb, bias_leb, X_test_leb, False)) 

    '''  Part 5 '''
    print('Dataset Lebkuchenstadt with data reuploading')
    weights_leb_re, bias_leb_re, metrics_leb_re = run_model('lebkuchenstadt', X_train_leb, y_train_leb, X_test_leb, y_test_leb, True)
    
    y_pred_leb_re = np.sign(prediction(weights_leb_re, bias_leb_re, X_test_leb, True))



    ''' Krampuskogel dataset '''

    #### Preprocessing
    X_kramp, y_kramp = data_preprocessing(df_kramp)

    ### Split data into train-test
    X_train_kramp, y_train_kramp, X_test_kramp, y_test_kramp = train_test_split(X_kramp, y_kramp)
    
    print('Dataset Krampuskogel without data reuploading')
    weights_kramp, bias_kramp, metrics_kramp = run_model('krampuskogel',  X_train_kramp, y_train_kramp, X_test_kramp, y_test_kramp, False)
    
    y_pred_kramp = np.sign(prediction(weights_kramp, bias_kramp, X_test_kramp, False))

    '''  Part 5 '''
    print('Dataset Krampuskogel with data reuploading')
    weights_kramp_re, bias_kramp_re, metrics_kramp_re = run_model('krampuskogel',  X_train_kramp, y_train_kramp, X_test_kramp, y_test_kramp, True)
    
    y_pred_kramp_re = np.sign(prediction(weights_kramp_re, bias_kramp_re, X_test_kramp, True))



    ''' Part 6 '''

    # Plot data 

    plot_accuracy(metrics_glueh, metrics_glueh_re, 'Accuracy over epoch for Gluehweindorf dataset')
    plot_accuracy(metrics_leb, metrics_leb_re, 'Accuracy over epoch for Lebkuchenstadt dataset')
    plot_accuracy(metrics_kramp, metrics_kramp_re, 'Accuracy over epoch for Krampuskogel dataset')


    plot_loss(metrics_glueh, metrics_glueh_re, 'Loss over epoch for Gluehweindorf dataset')
    plot_loss(metrics_leb, metrics_leb_re, 'Loss over epoch for Lebkuchenstadt dataset')
    plot_loss(metrics_kramp, metrics_kramp_re, 'Loss over epoch for Krampuskogel dataset')

    
    plot_confusion_matrix(y_test_glueh, y_pred_glueh, "Glühweindorf – Confusion Matrix (no re-upload)")

    plot_confusion_matrix(y_test_glueh, y_pred_glueh_re, "Glühweindorf – Confusion Matrix (re-upload)")

    plot_confusion_matrix(y_test_leb, y_pred_leb, "Lebkuchenstadt – Confusion Matrix (no re-upload)")

    plot_confusion_matrix(y_test_leb, y_pred_leb_re, "Lebkuchenstadt – Confusion Matrix (upload)")

    plot_confusion_matrix(y_test_kramp, y_pred_kramp, "Krampuskogel – Confusion Matrix (no re-upload)")

    plot_confusion_matrix(y_test_kramp, y_pred_kramp_re, "Krampuskogel – Confusion Matrix (upload)")



if __name__ == "__main__":

    run_exercise1()