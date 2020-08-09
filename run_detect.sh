cd codes/models/detect/EfficientDet
CUDA_VISIBLE_DEVICES=0 nohup python efficientdet_test_1.py>output_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python efficientdet_test_2.py>output_2.log 2>&1 &