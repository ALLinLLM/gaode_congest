# 天池高德比赛

### 提交记录

|模型|分数|备注|
|:--|:--|:--
|cos_sim|0.5155|VGG19(conv2d-19) + argmin(1-cos(a,b))
|vgg19feature_extract| - |过拟合

### data
参考数据目录`datasets/README.md`

### startup
1. git clone git@github.com:vegetable09/gaode_congest.git
2. `cd gaode_congest`
2. 按照`datasets/README.md`下载好数据和json到`datasets/`
2. `cd codes/data`
3. `sh get_Xy.sh`
4. 下载`https://download.pytorch.org/models/vgg19-dcbb9e9d.pth`到`~/.cache/torch/checkpoints/vgg19-dcbb9e9d.pth`
5. `sh run_model_cos.sh`
6. python embedding_distance_test.py
7. 结果保存在`datasets/amap_traffic_annotations_test_result.json`


### codes
所有方法在`codes/models`目录, 目前有

- cos_sim: 使用VGG19的卷积层获得每个分类的embedding中心, 对于test数据, 计算余弦距离, 取最近的
- vgg19feature_extract: 使用完整的VGG19, 接上softmax层, 过拟合