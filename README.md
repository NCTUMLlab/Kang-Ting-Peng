# Generative Adversarial Networks for Data Augmentation on Speaker Recognition
In this project, we implement three GANs model for data augmentation on speaker recognition.

You need to download NIST 2014 i-vector Machine Learning Challenge. Convert the data into numpy array and put it to NIST_npy/.

The dataset.py is used to handle the dataset  (ex. next_batch, sort, add), If everything is ready, execute main.py to start training.


# Architecture
<img src="fig/GAN-Cos-Gen.png" width = 50% height = 50% alt="GAN-Cos-Gen" align=center />

# Setting
- Python 2.7.12
- Tensorflow 1.1
