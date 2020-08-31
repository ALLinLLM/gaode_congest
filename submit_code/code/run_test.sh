# 目标检测结果放在../user_data下，如果需要重新跑，取消下面的注释
cd detection
CUDA_VISIBLE_DEVICES=0 nohup python run_detection_for_train.py >0.log 2>&1 & echo $! > pid_0.txt
# CUDA_VISIBLE_DEVICES=1 nohup python run_detection_for_test.py >1.log 2>&1 & echo $! > pid_1.txt
# CUDA_VISIBLE_DEVICES=2 nohup python run_detection_for_testb.py >2.log 2>&1 & echo $! > pid_2.txt
# cd ..
# python main.py