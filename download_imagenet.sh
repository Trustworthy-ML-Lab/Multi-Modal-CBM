cd data
mkdir imagenet
cd imagenet

wget #ImageNet training set link address
mkdir -p train
tar -xvf ILSVRC2012_img_train.tar -C ./train

dir=./train 
for x in `ls $dir/*tar`
do	
  filename=`basename $x .tar`     
  mkdir $dir/$filename     
  tar -xvf $x -C $dir/$filename 
done 
rm *.tar

wget #ImageNet validation set link address
mkdir -p val
tar -xvf ILSVRC2012_img_val.tar -C ./val
cd val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh > valprep.sh
chmod +x valprep.sh
./valprep.sh
rm valprep.sh
cd ..