DIR="./TextClassification"
mkdir $DIR
cd $DIR

rm -rf SST-2
wget --content-disposition https://cloud.tsinghua.edu.cn/f/bccfdb243eca404f8bf3/?dl=1
tar -zxvf SST-2.tar.gz
rm -rf SST-2.tar.gz

cd ..