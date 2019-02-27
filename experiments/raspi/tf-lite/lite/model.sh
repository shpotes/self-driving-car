rm -rf model
mkdir model
cd model
wget http://download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz
tar xvf inception_v4_299_quant_20181026.tgz
rm inception_v4_299_quant_20181026.tgz
