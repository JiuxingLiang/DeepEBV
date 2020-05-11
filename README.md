# DeepEBV
DeepEBV was designed for detection of EBV virus in DNA integration by deep learning.

Run ‘Data_Process.py’ to get one-hot encoding test data and labels.
Run ‘DeepEBV_Test.py’ to detect DNA sequence of EBV virus.

Framework:
DeepEBV framework model contains input layer, 1st convolution1D layer, 1st max pooling layer, 2nd convolution1D layer, 2nd max pooling layer, 1st dropout layer, 2nd dropout layer, attention layer, 1st dense layer (fully connected layer), 2nd dense layer, concatenate layer, classifier layer.

Dependency:
Keras library 2.2.4. 
scikit-learn 0.22. 

If you have any questions, please contact me.

Email: liangjiuxing@m.scnu.edu.cn
