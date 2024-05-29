"""
Main code to run the neural networks designed in fc_nn.py and cnn.py
Source: Stanford CS231n course materials, modified by Sara Mathieson
Micah Painter, Elle McMahon
Date: 05/02/2024
"""

# imports
from cnn import CNNmodel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from util import load_data
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def main():

    # Choose path based on who is running the code, mpainter path seems to be public, so use that if in doubt
    #train_path_name = "/homes/emcmahon/data/ASL_Alphabet_Dataset/asl_alphabet_train"
    train_path_name = "/homes/mpainter/cs360/project/archive/ASL_Alphabet_Dataset/asl_alphabet_train"
    test_path_name = "/homes/mpainter/cs360/project/archive/ASL_Alphabet_Dataset/asl_alphabet_test"
    X_train, y_train = load_data(train_path_name, True) # use the function in util.py to load the X and y data
    X_test, y_test = load_data(test_path_name, False)

    # Make sure that X and y data are in the correct format (float32)
    X_train = tf.cast(X_train, tf.float32)
    y_train = tf.cast(y_train, tf.float32)

    X_test = tf.cast(X_test, tf.float32)
    y_test = tf.cast(y_test, tf.float32)
    
    print("X_train: ", X_train)
    print("y_train: ", y_train)
    print("X_test: ", X_test)
    print("y_test:", y_test)
    print("X train shape: ", X_train.shape)
    print("y train shape: ", y_train.shape)
    print("X test shape: ", X_test.shape)
    print("y test shape: ", y_test.shape)

    # test_path_name = '/homes/mpainter/cs360/project/archive/ASL_Alphabet_Dataset/asl_alphabet_test'
    # X_test, y_test = load_data(test_path_name)

    # Get train and test data into tensor batches, shuffling train data
    train_dset = tf.data.Dataset.from_tensor_slices((X_train,y_train)).shuffle(10000).batch(64)
    test_dset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)

    # train_accuracy_FC, val_accuracy_FC, labels_actual_train_FC, labels_prediction_train_FC =  run_training(FCmodel(), train_dset,train_dset)

    # Call run_training on the training and testing dataset to get accuracy and labels
    train_accuracy_cnn, val_accuracy_cnn, all_val_labels, all_val_predictions, all_train_labels, all_train_predictions =  run_training(CNNmodel(), train_dset,test_dset)

    # print("train_accuracy: ", train_accuracy_FC)
    # print("test accuracy: ", val_accuracy_FC)
    # confusion_matrix_maker(labels_actual_train_FC, labels_prediction_train_FC, 'FC')
    # x=[0,1,2,3,4,5,6,7,8,9]
    # plot_curves(x,train_accuracy_FC)

    # print the final accuracy, make a confusion matrix, and plot an accuracy curve
    # print("train_accuracy: ", train_accuracy_cnn)
    # print("test accuracy: ", val_accuracy_cnn)
    # print("val labels: ", all_val_labels)
    # print("val pred: ", all_val_predictions)
    # print("train labels: ", all_train_labels)
    # print("train pred: ", all_train_predictions)
    confusion_matrix_maker(all_val_labels, all_val_predictions, 'test')
    confusion_matrix_maker(all_train_labels, all_train_predictions, 'train') # make another one of these
    x=[0,1,2,3,4,5,6,7,8,9] # number of epochs
    plot_curves(x,train_accuracy_cnn, "train")
    plot_curves(x, val_accuracy_cnn, "test")
    pass


def run_training(model, train_dset, val_dset):
    ''' 
    Given model, train and val set, run the training and validation,
    returns train accuracy list, val accuracy list, all true labels and  all predicted labels.
    '''

    #creating empty lists
    train_accuracy_list = []
    val_accuracy_list =[]
    all_val_predictions = []
    all_val_labels = []
    all_train_predictions = []
    all_train_labels = []

    # set up a loss_object (sparse categorical cross entropy)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    # use the Adam optimizer
    optimizer = tf.keras.optimizers.Adam()

    # set up metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    # make an array that has all the predictions for every single value for every single epoch
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
    for epoch in range(10):
        for images_train, labels_train in train_dset:
            # run train step to get the loss and prediction for a single value
            loss_train, prediction_train = train_step(model, images_train, labels_train, optimizer, loss_function)
            loss_train = train_loss(loss_train)
            accuracy_train = train_accuracy(labels_train, prediction_train)
            # all_train_predictions = all_train_predictions + [pd.Series(prediction_train).idxmax()]
            # all_train_labels.append(labels_train)
            # this is for finding out the predictions and labels so we can print the confusion matrix (only use the last epoch)
            if epoch == 9:
                for entry in prediction_train:
                    all_train_predictions = all_train_predictions + [pd.Series(entry).idxmax()] # find the max (our actual prediction)
                for entry in labels_train:
                    # get entry into the correct form to add to the array
                    entry = entry.numpy()
                    entry = int(entry)
                    all_train_labels.append(entry)

        print("train accuracy: ", accuracy_train)



    # loop over validation data and compute val_loss, val_accuracy
        for images_val, labels_val in val_dset:
             # run the validation step to get the loss and prediction for a single value
             loss_val, prediction_val = val_step(model, images_val, labels_val, loss_function)

             val_loss(loss_val)
             val_accuracy(labels_val, prediction_val)

             # this is for finding out the predictions and labels so we can print the confusion matrix (only use the last epoch)
             if epoch == 9:
                for entry in prediction_val:
                    all_val_predictions = all_val_predictions + [pd.Series(entry).idxmax()] # find the max (our actual prediction)
                for entry in labels_val:
                    # get entry into the correct form to add to the array
                    entry = entry.numpy()
                    entry = int(entry)
                    all_val_labels.append(entry)

        # add the training and validation accuracy to their respective lists
        train_accuracy_list.append(train_accuracy.result()*100)
        val_accuracy_list.append(val_accuracy.result()*100)

        print(template.format(epoch+1,
                            train_loss.result(),
                            train_accuracy.result()*100,
                            val_loss.result(),
                            val_accuracy.result()*100))

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        print("all labels: ", all_val_labels)
        print("all predictions: ", all_val_predictions)


    return train_accuracy_list, val_accuracy_list, all_val_labels, all_val_predictions, all_train_labels, all_train_predictions

def train_step(model, images, labels, optimizer, loss_function): 
    '''
    Trains the model on one input.
    
    Input the desired model (in this case either cnn or fc_nn, a single image 
    (an x point), a label (y point), and an optimizer and loss_function defined in main.

    Outputs the loss and prediction for that one datapoint
    '''

    images = tf.cast(images, tf.float32)

    with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
        # compute the predictions given the images, then compute the loss
        prediction = model(images, training=True)
        loss = loss_function(labels, prediction)
    
    # compute the gradient with respect to the model parameters (weights)
    gradients = tape.gradient(loss, model.trainable_variables)
    # apply this gradient to update the weights (i.e. gradient descent)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # return the loss and predictions
    return loss, prediction

def val_step(model, images, labels, loss_function): 
    ''' 
    Given model, images, label and loss_function compute preidctions and 
    loss for validation, returns, loss and prediction.
    '''
  
    # compute the predictions given the images, then compute the loss
    prediction = model(images, training=False)
    loss = loss_function(labels, prediction)


    # return the loss and predictions
    return loss, prediction

def confusion_matrix_maker(actual_labels, predicted_labels, model):
    '''
    Method to make the confusion matrix given actual/ predicted labels and the model
    returns the confusion matrix
    '''
    #making labels
    real_labels = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'] 
    labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    #labels = [24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
    #real_labels = ['a','b','c','d']
    #labels = [0,1,2,3]
    #creating confusion matrix
    confusion_matrixx = confusion_matrix(actual_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrixx,display_labels= real_labels)
    disp.title = ('Confusion Matrix for ' + model)
    disp.plot()
    plt.xticks(rotation = 90) # change the direction of the xlabels to verticle
    plt.savefig('figures/Confusion_Matrix_'+model+'.jpg', bbox_inches='tight')
    plt.show()

def plot_curves(x,train_fc_acc, model):
    '''
    Method to plot the curves, given all accuracies
    plots and saves the figure.
    '''
    plt.clf()
    #plt.plot(x, train_cnn_acc, label = "CNN training accuracy")
   #plt.plot(x, val_cnn_acc, label = "CNN validation accuracy")
    plt.plot(x, train_fc_acc, label = "FC training accuracy")
    #plt.plot(x, val_fc_acc, label = "FC validation accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy by %')
    plt.legend()
   #plt.title("Accuracy: FN vs. CNN")
    plt.title("Accuracy by epoch FC")
    # fig_save = ('figures/Graph.pdf')
    fig_save = ('figures/Graph_'+model+'.jpg')
    plt.savefig(fig_save)
    plt.show()
    plt.clf()

main()
