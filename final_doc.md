# Weka源码阅读
[TOC]

## 1. Weka简介及使用样例展示
### Weka简介
xxxx
###使用Weka
我们总结了使用weka的四种方式。
####图形界面
####命令行调用

```
java -cp WEKA_INS/weka.jar weka.classifiers.functions.Logistic 
-t WEKA_INS/data/weather.numeric.arff 
-T WEKA_INS/data/weather.numeric.arff 
-d ./weather.numeric.model.arff
```
其中，`-t`用于设置训练集，`-T`设置测试集，`-d`？


####java调用
####其他语言的接口调用


## 2. Weka的框架，核心类


## 3. Weka中的算法实现


## 4. Weka中的工具类

## 参考文献
[1] http://stats.stackexchange.com/questions/71684/how-to-interpret-weka-logistic-regression-output