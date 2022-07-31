set -x

audio_path="/home/zhanghm/Research/V100/TalkingFaceFormer/data/audio_samples/Silence.wav"
flist="WDA_ChrisMurphy0_000" # ./datasets/test_WDA_ChrisMurphy0_000.txt

python test_face2vid.py --name HDTF_WDA_ChrisMurphy0_000_wo_noise_localG_512p_lower_neck \
                --dataset_name FaceDataset \
                --dataroot ./datasets/HDTF_face3dmmformer/train \
                --flist ${flist} \
                --batchSize 1 \
                --no_instance --label_nc 0 --resize_or_crop none --input_nc 6 \
                --netG local --loadSize 512 --ngf 32 \
                --which_epoch latest --how_many 200 --serial_batches \
                --audio_path ${audio_path} \
                --input_rendered_face /home/zhanghm/Research/V100/TalkingFaceFormer/test_dir/AAAI/test_WDA_ChrisMurphy0_000-Silence_audio