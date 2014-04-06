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


预测结果
--------

真实购买记录一共有3526条


TODO
----

0. *注意*调整正负样本比例

1. 尝试把LR做成二次方的，即用二次曲线拟合数据。

1. 在LR的基础上做MSLR，样本提取要更加合理。

1. 在LR的基础上做RawLR。按照天猫内部的思路来。

2. 在UserCF和ItemCF上加上时间因子的影响。

3. 利用UserCF做好的用户聚类、ItemCF做好的品牌聚类来做细化的LR，或者在聚类
上做LFM

4. 在ItemCF的思路上挖掘频繁项集/购买模式，如购买品牌A和商品后往往会购买
品牌B的商品

5. LFM


参考论文
--------

See https://www.google.com.hk/search?q=data+mining+time+series&ie=utf-8&oe=utf-8&aq=t for more.

Chapter 1 MINING TIME SERIES DATA - ResearchGate


模型列表
----------

LR(model=LinearSVC(C=10, loss='l1'), alpha=0.7, degree=1)

    |         TOTAL   VISITED BOUGHT  FAVO    CART    NEW
    | Pred #  1438    1436    626     71      12
    |      %  100%    99.861% 43.533% 4.937%  0.834%
    | Real #  1311    250     89      10      1
    |      %  100%    19.069% 6.789%  0.763%  0.076%
    Hit #  76
    Precision  5.285118%
    Recall     5.797101%
    F1 Score   5.529283%

这个模型在数据标成变成2次后，Precision ~ 16%，同时F1 ~ 3%


LR(model=LogisticRegression(penalty='l1'), alpha=0.7, degree=1)

    |         TOTAL   VISITED BOUGHT  FAVO    CART    NEW
    | Pred #  1472    1470    615     68      14
    |      %  100%    99.864% 41.780% 4.620%  0.951%
    | Real #  1311    250     89      10      1
    |      %  100%    19.069% 6.789%  0.763%  0.076%
    Hit #  74
    Precision  5.027174%
    Recall     5.644546%
    F1 Score   5.318002%

这个模型在数据标成变成2次后，Recall > 8%，同时F1 > 5%


LR(model=Perceptron(penalty='l1'), alpha=0.7, degree=1)

    |         TOTAL   VISITED BOUGHT  FAVO    CART    NEW
    | Pred #  3145    3140    1023    130     26
    |      %  100%    99.841% 32.528% 4.134%  0.827%
    | Real #  1311    250     89      10      1
    |      %  100%    19.069% 6.789%  0.763%  0.076%
    Hit #  113
    Precision  3.593005%
    Recall     8.619375%
    F1 Score   5.071813%


LR(model=PassiveAggressiveClassifier(C=1, loss='hinge'), alpha=0.7, degree=1)

    |         TOTAL   VISITED BOUGHT  FAVO    CART    NEW
    | Pred #  2608    2603    823     119     22
    |      %  100%    99.808% 31.557% 4.563%  0.844%
    | Real #  1311    250     89      10      1
    |      %  100%    19.069% 6.789%  0.763%  0.076%
    Hit #  98
    Precision  3.757669%
    Recall     7.475210%
    F1 Score   5.001276%


model2, PassiveAggressiveClassifier(C=1, loss='squared_hinge')

    |         TOTAL   VISITED BOUGHT  FAVO    CART    NEW
    | Pred #  5172    5161    1408    203     29
    |      %  100%    99.787% 27.224% 3.925%  0.561%
    | Real #  1311    250     89      10      1
    |      %  100%    19.069% 6.789%  0.763%  0.076%
    Hit #  129
    Precision  2.494200%
    Recall     9.839817%
    F1 Score   3.979639%
