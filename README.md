
Names: Micah Painter and Elle McMahon

Description:
Uses images of ASL alphabet signs to train a Convolutional Neural Network.
Uses different images of ASL alphabet signs to test the network, and then creates accuracy curves and confusion matrices.
Uses images of all the letters except j and z, as those include movement.

To Run:
Run the file 'run_cnn.py' as you would any other file. No additional command line arguments needed
In run_cnn, change train_path_name and test_path_name to be the paths that lead to the data.
The uncommented path seems to be public, so use that if in doubt.

Data Format:
The data (both training and test) must be divided into folders labeled by the letter that represents the label.
Images must be .jpg

Reproducability:
There is some randomness here, most notably in the selection of images used for the training data. 
As such, results will vary slightly on different runs.
See our write-up for our results.
