set -x

python train.py --name HDTF_face3dmmformer_with_noise_localG --dataset_name FaceDataset --dataroot ./datasets/HDTF_face3dmmformer/train \
                --flist ./datasets/train_HDTF_face3dmmformer.txt --batchSize 1 \
                --no_instance --label_nc 0 --resize_or_crop none --input_nc 6 --netG local