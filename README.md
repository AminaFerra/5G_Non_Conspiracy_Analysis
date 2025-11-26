# 5G_Non_Conspiracy_Analysis
Accuracy Improvements Summary:

To improve the GCN model’s accuracy on the Cora dataset:

1- Increased the hidden layer size from 4 to 16 to allow more node feature representation.

2- Added a dropout layer with a 0.5 probability after the first GCN layer to prevent overfitting.

3- Added weight decay of 5e-4 in the Adam optimizer.

4- Increased the number of training epochs from 50 to 200.