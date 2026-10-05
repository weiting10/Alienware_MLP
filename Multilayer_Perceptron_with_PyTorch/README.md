![alt text](https://miro.medium.com/max/874/0*HB2hrDZDqs8l5KFA.jpeg)

*Image source: Internet*

Multilayer perceptron or simple 3 layer neural network is implemented using Pytorch and Sklearn frameworks. The data set is MNIST (hand writtem digits 0-9).

**Required packages**
 1. Python 3.8 or higher (with numpy)
 2. Pytorch (and torchvision)
 3. Sklearn
 4. Matplotlib

**The script follow steps:**
1. Data pre-processing including normalizing and preprocessing with augmentation
2. Defining Multilayer perceptron model
3. Optimizer and cost function
4. Training loop and testing
5. Results visualization (Accuracy, confusion matrix, represntations with dimensionality reduction and more)

**Usage**

The script is originally written for 10 Epochs, one can change it and achive higher results or make the model overfit at somepoint. 

Simply run the file
```
Multilayer_Perceptron.py
```
**Results visualizations**

Apart from usual performance measurements such as test and validation accuracy

1. confusion matrix ![alt text](https://github.com/DashankaNadeeshanDeSilva/Multilayer_Perceptron/blob/main/confusion_matrix.png)
2. Plot of incorrectly predicted images along with how confident they were on the actual label ![alt text](https://github.com/DashankaNadeeshanDeSilva/Multilayer_Perceptron/blob/main/most_incorrect.png)
3. output and intermediate representations from the model after applying PCA for dimensionality reduction ![alt text](https://github.com/DashankaNadeeshanDeSilva/Multilayer_Perceptron/blob/main/intermediates_representations.png)
4. plot of the weights in the first layer of the model ![alt text](https://github.com/DashankaNadeeshanDeSilva/Multilayer_Perceptron/blob/main/weights_visualization.png)
