#!/bin/bash


FILE_ID_1="14FrmMPQ2C9np-mHsIdF47Pj5V3pO8Ioq"
FILE_ID_2="1QlKBlHldUZx-1v1Awe0jfuEH6x_Rjj6N"
FILE_ID_3="1m9kla12byCgJNp8N5xP22np1j0q4adeG"

gdown https://drive.google.com/uc?id=$FILE_ID_1 -O BYOL_pretrained_weight.pt
echo "Downloaded model1.pth"

gdown https://drive.google.com/uc?id=$FILE_ID_2 -O p1_finetune_weight.pth
echo "Downloaded model2.pth"

gdown https://drive.google.com/uc?id=$FILE_ID_3 -O p2_weight.pth
echo "Downloaded model3.pth"

echo "All downloads complete."
