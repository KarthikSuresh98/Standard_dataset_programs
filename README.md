# Standard_dataset_programs
Deep learning models implemeted on standard Keras and Kaggle datasets. Different network architectures were implemented
on these datasets resulting in good testing accuracy in each of the cases

Datasets used are:
 1. Cifar10 dataset
 2. Mnist dataset
 3. Fashion Mnist dataset
 4. Parkinson's kaggle dataset

Note:

The resnet.py is a program wherein the concept of residual networks are explored. Resnets always tend to perform better than 
any normal network achitecture. In this case the resnet is implemented on the yalefaces dataset. The program doesn't give good 
results since yalefaces is very small dataset with only 165 images thus causing the ntwork to perform poorly. But these
networks when tested on above datasets like cifar10 and mnist always yield better testing accuracy.
