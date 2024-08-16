# crisprDBA

We propose CRISPR-DBA, a Bayesian-based neural network framework to address this challenge. Instead of introducing new parameters to simulate model uncertainty, the proposed method is the first neural network-based machine learning model that utilizes dropout layer information to extract distribution of off-target cleavage activities for CRISPR/Cas gRNAs. It also outperforms an existing probabilistic model in confidence performance. The three key features of the proposed method include 1) the ability to generate trustworthy predictions by providing extra confidence readings of prediction; 2) reduction of complexity compared to approaches equipped with similar functionality; 3) adaptability in terms of being able to utilise existing model architectures developed in the field.

PREREQUISITE
------------
The PNN off-target prediction models were conducted using Python 3.8.8, Keras 2.4.3, and TensorFlow v2.3.0. The following Python packages should be installed:

scipy 1.7.1

numpy 1.18.5

pandas 1.3.2

scikit-learn

TensorFlow


