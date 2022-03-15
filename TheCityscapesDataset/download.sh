#!/bin/bash -x
wget --keep-session-cookies --save-cookies=cookies.txt --post-data "username=$USERNAME&password=$PASSWORD&submit=Login" https://www.cityscapes-dataset.com/login/

wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3

unzip -qn gtFine_trainvaltest.zip
unzip -qn leftImg8bit_trainvaltest.zip

mkdir test_img test_inst test_label
mkdir train_img train_inst train_label

cp gtFine/test/*/*inst* test_inst/
cp gtFine/test/*/*label* test_label/
cp leftImg8bit/test/*/*Img8bit* test_img/

cp gtFine/train/*/*inst* train_inst/
cp gtFine/train/*/*label* train_label/
cp leftImg8bit/train/*/*Img8bit* train_img/

rm index.html* cookies.txt README license.txt
rm gtFine_trainvaltest.zip
rm leftImg8bit_trainvaltest.zip
rm -r gtFine
rm -r leftImg8bit