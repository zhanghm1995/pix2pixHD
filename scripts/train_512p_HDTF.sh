set -x

python train.py --name HDTF_512p --dataset_name FaceDataset --dataroot ./datasets/HDTF_preprocessed \
                --flist ./datasets/train.txt --batchSize 2 \
                --no_instance --label_nc 0