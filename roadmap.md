# RoadMap

## TODO
现在可以做的是

1. 用weka自带的数据集及它的图形化界面跑一下（这个是每个人都必须做的），这种网上有很多的教程
2. 按[知乎](http://www.zhihu.com/question/22401599)上的这个人的回答做一遍。应该要输出一个函数跳转过程的图。
3. 看分类器的代码。
4. 看聚类方法的代码。
5. 看评价方法、其他工具函数的代码。
6. 一些影响性能、速度的关键代码。

看代码的输出可以参考后面参考文献中的weka源码分析，看代码之前先把源码分析中对应的部分看一下。

## 基础
Instances
Classifier

## 分类器

ZeroR
OneR

Id3   总裁
J48

LinearRegression  吴
LogisticRegression 吴
MultiLayerPerceptron 吴
SGD 吴


Knn(IBk) 钱

Bagging
Adaboost

SVM 

NaiveBayes 闫


**LWL**


## 无监督
SimpleKMeans  

EM
PCA

DBSCAN
OPTICS

## 评价
Evaluation

## 关联规则
Apriori

## 影响性能速度的关键代码
**StringToWordVector** 吴

## Weka Python Wrapper

## References
 [1] http://quweiprotoss.blog.163.com/blog/#m=0&t=1&c=fks_087064082095085064093081086095080086089075086094094064, weka源码分析`网上流传的weka源码分析的原始出处`
 
 [2] https://weka.wikispaces.com/Use+WEKA+in+your+Java+code, Use Weka in your Java code