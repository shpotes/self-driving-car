rm -rf model
mkdir model
cd model
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz
tar xvf inception_resnet_v2_2018_04_27.tgz
rm inception_resnet_v2_2018_04_27.tgz
