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

而我在数据预处理时将其重新标记为：

    点击：1
    购买：4
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
