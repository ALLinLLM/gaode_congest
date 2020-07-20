# 天池高德比赛

### 提交记录

|模型|分数|备注|
|:--|:--|:--
|cos_sim|0.5155|VGG19(conv2d-19) + argmin(1-cos(a,b))
|vgg19feature_extract| - |过拟合

### data
参考数据目录`datasets/README.md`


### codes
所有方法在`codes/models`目录, 目前有

- cos_sim: 使用VGG19的卷积层获得每个分类的embedding中心, 对于test数据, 计算余弦距离, 取最近的
- vgg19feature_extract: 使用完整的VGG19, 接上softmax层, 过拟合