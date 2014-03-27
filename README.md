AliDMCompetition
================

阿里巴巴大数据竞赛（http://102.alibaba.com/competition/addDiscovery/index.htm ）


数据说明
--------

提供的原始文件有大约4M左右，涉及1千多天猫用户，几千个天猫品牌，总共10万多条的行为记录。

用户4种行为类型(Type)对应代码分别为：

    点击：0
    购买：1
    收藏：2
    购物车：3


提交格式
--------

参赛者将预测的用户存入文本文件中，格式如下：

```user_id \t brand_id , brand_id , brand_id \n```

上传的结果文件名字不限(20字以内)，文件必须为txt格式。


参考论文
--------

See https://www.google.com.hk/search?q=data+mining+time+series&ie=utf-8&oe=utf-8&aq=t for more.

Chapter 1 MINING TIME SERIES DATA - ResearchGate


Model List
----------

model2, LinearSVC(C=10, loss='l1')

    Precision: 0.046473
    Recall:    0.063433
    F1 Score:  0.053645


model2, LogisticRegression(penalty='l1')

    Precision: 0.049242
    Recall:    0.058209
    F1 score:  0.053352


model2, Perceptron(penalty='l1')

    Precision: 0.039648
    Recall:    0.080597
    F1 score:  0.053150


model2, PassiveAggressiveClassifier(C=1, loss='hinge')

    Precision: 0.044615
    Recall:    0.064925
    F1 score:  0.052888


model2, PassiveAggressiveClassifier(C=1, loss='squared_hinge')

    Precision: 0.025883
    Recall:    0.094030
    F1 score:  0.040593
