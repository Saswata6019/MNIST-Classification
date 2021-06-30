# MNIST Classification
Digit Classification using the MNIST dataset and Image Processing (Tensorflow 1.12.0 and Python 3.6.6)

- Editor used: Sublime Text 3
- Shell used to run the code: Git Bash
- Libraries used: Numpy, Matplotlib, PIL & Tensorflow
- The MNIST dataset is downloaded using the load_data() function

Insight on iris.csv
- There are a total of 5 columns of data
- The first 4 columns serve as the features for the model
- The last (5th) column serves as the result, which the model predicts and trains itself on during the prediction and training+testing stages respectively. The iris_predict.csv file does not contain the 5th/result column since the model is supposed to predict that result and generate the same as it's output.
- Column 1: Represents the sepal length of an individual flower
- Coumn 2: Represents the sepal width of an individual flower
- Column 3: Represents the petal length of an individual flower
- Column 4: Represents the petal width of an individual flower
- Column 5: Represents the species of an individual flower
