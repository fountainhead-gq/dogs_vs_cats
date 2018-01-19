# Cats vs Dogs

猫狗识别的数据集来自Kaggle，[dogs_vs_cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)



### 依赖
- GPU on AWS

- Python 3.5

- Tensorflow

- Keras

- sklearn

- jupyter notebook

  ​


### 模型

- Xception 

- VGG16 

- ResNet50 

- InceptionV3 

- InceptionResNetV2

  ​



### 改进

- Fine tune

- 多模型融合

  ​



### 结果

| model                                    | epochs | LogLoss | ranking |
| ---------------------------------------- | ------ | ------- | ------- |
| InceptionV3                              | 10     | 0.07607 | >220    |
| Xception                                 | 10     | 0.05390 | 90      |
| inceptionResNetV2                        | 10     | 0.06508 | <150    |
| Xception(finetune)                       | 8      | 0.04205 | <20     |
| InceptionV3 + Xception + inceptionResNetV2 | 10     | 0.03826 | <12     |



### 参考

- [ResNet](https://arxiv.org/abs/1512.03385)
- [InceptionV3 ](https://arxiv.org/abs/1512.00567)
- [Xception](https://arxiv.org/abs/1610.02357)
- [logloss](https://www.kaggle.com/wiki/LogLoss)
- [keras](https://keras.io/)
- [dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats)


