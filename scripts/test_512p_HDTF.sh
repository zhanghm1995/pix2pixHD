set -x

python test_face2vid.py --name HDTF_WDA_ChrisMurphy0_000_wo_noise_localG_512p --dataset_name FaceDataset \
                --dataroot ./datasets/HDTF_face3dmmformer/val \
                --flist ./datasets/test_WDA_ChrisMurphy0_000.txt --batchSize 1 \
                --no_instance --label_nc 0 --resize_or_crop none --input_nc 6 \
                --netG local --loadSize 512 --ngf 32 \
                --which_epoch latest --how_many 200 --serial_batches \
                --audio_path datasets/HDTF_face3dmmformer/val/WDA_ChrisMurphy0_000/WDA_ChrisMurphy0_000.wav