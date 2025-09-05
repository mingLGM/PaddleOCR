
##——————训练——————##

##定位
python tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml -o Global.pretrained_model=./pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams

##识别
python tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml -o Global.pretrained_model=./pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams





##——————评估——————##

python tools/infer_det.py  -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml -o Global.infer_img="./datasets/general_ocr_002.png" Global.checkpoints="./output/PP-OCRv5_mobile_det/latest"

##定位
python tools/eval.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml -o Global.pretrained_model=output/PP-OCRv5_mobile_det/latest.pdparams Eval.dataset.data_dir=./train_data/icdar2015/text_localization/ Eval.dataset.label_file_list=./train_data/icdar2015/text_localization/test_icdar2015_label.txt

##识别



##——————导出——————##

##定位
python tools/export_model.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml -o Global.pretrained_model=output/PP-OCRv5_mobile_det/latest.pdparams Global.save_inference_dir="./output/PP-OCRv5_mobile_det/PP-OCRv5_mobile_det_infer/"





###############################
####____gangban____####
CUDA_VISIBLE_DEVICES=1 python -m paddle.distributed.launch --gpus '1' tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det_gangban.yml
CUDA_VISIBLE_DEVICES=0 python -m paddle.distributed.launch --gpus '0' tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec_gangban.yml

python tools/infer_det.py  -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det_gangban.yml -o Global.infer_img="/home/zhangshuwen/work/datasets/gangban/paddleocr_dataset/det/images/00000_20250801_144146_1_CAM1_NG_Watermark1.bmp" Global.checkpoints="./output/gangban/PP-OCRv5_mobile_det/best_model/model"
python tools/eval.py -c configs/det/PP-OCRv5/PP-OCRv5_mobile_det_gangban.yml -o Global.pretrained_model=./output/gangban/PP-OCRv5_mobile_det/best_model/model.pdparams Eval.dataset.data_dir=/home/zhangshuwen/work/datasets/gangban/paddleocr_dataset/det/ Eval.dataset.label_file_list=/home/zhangshuwen/work/datasets/gangban/paddleocr_dataset/det/det_label.txt
python tools/infer/predict_system.py --image_dir="/home/zhangshuwen/work/datasets/gangban/paddleocr_dataset/det/images/" --det_model_dir="./output/gangban/PP-OCRv5_mobile_det/PP-OCRv5_mobile_det_infer/" --rec_model_dir="./output/gangban/PP-OCRv5_mobile_rec/PP-OCRv5_mobile_rec_infer/" --use_angle_cls=false


python tools/infer_rec.py  -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec_gangban.yml -o Global.infer_img="/home/zhangshuwen/work/datasets/gangban/paddleocr_dataset/rec/images/" Global.checkpoints="./output/gangban/PP-OCRv5_mobile_rec/latest"


python tools/export_model.py -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec_gangban.yml -o Global.pretrained_model="./output/gangban/PP-OCRv5_mobile_rec/best_model/model.pdparams" Global.save_inference_dir="./output/gangban/PP-OCRv5_mobile_rec/PP-OCRv5_mobile_rec_infer/"

