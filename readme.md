# Extreme Learning Autoencoder Tensorflow r1.13 implementation
---
##### This code is a tensorflow implementation of the algorithm on the paper: `Autoencoder with Invertible Functions for Dimension Reduction and Image Reconstruction`
##### Authors of the paper:  Yimin Yang, Jonathan Wu, and Yaonan Wang

---
Author of this code: Peizhi Yan
Affiliation: Lakehead University             
Personal Website: https://PeizhiYan.github.io
Date: March 24th, 2019                       
Tensorflow version: r1.13

---

#### Instruction:

For example, assume you want to reduce the dimension of MNIST dataset from 28*28 to 100, you can use the following lines of code:
```python
"""create the ELA model"""
model = ELA(input_units=28*28, hidden_units=100, c=2**(8), activation='sin') 

"""train the model"""
model.train(x_train, 100) # x_train is the training dataset with shape [50000,28*28]; 100 is the numer of total training epochs

"""evaluate the model"""
print("MSE(train) :", model.evaluate(x_train))
print("MSE(train) :", model.evaluate(x_test))

"""encode data"""
x_train_encoding = model.encoding(x_train)
```
