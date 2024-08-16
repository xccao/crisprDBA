# crisprDBA

CRISPR-DBA is a Bayesian-based neural network framework designed to address the challenge of off-target effects that may arise from the low specificity of certain guide RNAs (gRNAs). 

Instead of introducing new parameters to simulate model uncertainty, this method represents the first neural network-based machine learning model that utilizes dropout layer information to extract the distribution of off-target cleavage activities for CRISPR/Cas gRNAs. 

The CRISPR-DBA method offers three key advantages: 
1) it generates reliable predictions by providing additional confidence estimates;
2) it reduces complexity compared to other approaches with similar functionality;
3) it is adaptable, as it leverages existing model architectures developed within the field. 

PREREQUISITE
------------
The PNN off-target prediction models were conducted using Python 3.8.8, Keras 2.4.3, and TensorFlow v2.3.0. The following Python packages should be installed:

scipy 1.7.1

numpy 1.18.5

pandas 1.3.2

scikit-learn > 1.1.3

TensorFlow > 2.3.0

USAGE
------------
An example of crisprDBA can be run as: 

```
python FNN5_PNN.py crisprSQL_723_format.npz
```


