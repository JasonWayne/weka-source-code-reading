# Weka源码阅读
[TOC]

## 1. Weka简介及使用样例展示
### Weka简介
xxxx
#### 本节参考文献


###使用Weka
我们总结了使用weka的四种方式，下面一一介绍。
####图形界面
这应该是最常用的方式，也是Weka流行的原因。
####命令行调用

```
java -cp WEKA_INS/weka.jar weka.classifiers.functions.Logistic 
-t WEKA_INS/data/weather.numeric.arff 
-T WEKA_INS/data/weather.numeric.arff 
-d ./weather.numeric.model.arff
```
其中，`WEKA_INS`为Weka的安装地址，`-t`用于设置训练集，`-T`设置测试集，这里分别用Weka自带的天气数据作为训练和测试集，`-d`？


####java调用
####其他语言的接口调用

#### 本节参考文献
[1] http://stats.stackexchange.com/questions/71684/how-to-interpret-weka-logistic-regression-output


## 2. Weka的框架，核心类
### Weka的输入
#### .arff格式
Weka希望用户以arff格式作为输入（现在也支持CSV和其他格式，arff是Attribute-Relation File Format的缩写。下面是一个arff文件的示例，前面几行类似于定义数据库的表名，字段名和字段类型，`@DATA`后面的内容是真正的数据，csv的格式。

```
@RELATION house

@ATTRIBUTE houseSize NUMERIC
@ATTRIBUTE lotSize NUMERIC
@ATTRIBUTE bedrooms NUMERIC
@ATTRIBUTE granite NUMERIC
@ATTRIBUTE bathroom NUMERIC
@ATTRIBUTE sellingPrice NUMERIC

@DATA
3529,9191,6,0,0,205000 
3247,10061,5,1,1,224900 
4032,10150,5,0,1,197900 
2397,14156,4,1,0,189900 
2200,9600,4,0,1,195000 
3536,19994,6,1,1,325000 
2983,9365,5,0,1,230000
```
#### Instances类

### Weka中的核心类
对于一个机器学习的任务，一般都有数据预处理，训练模型，评价这三个步骤，在Weka中，均分别有对应的类，下面以分类问题为例，详细介绍在解决一般的分类问题过程中都会用到的核心类。
#### Filter
#### AbstractClassifier

#### Evaluation

#### 本节参考文献
[1] http://www.ibm.com/developerworks/library/os-weka1/




## 3. Weka中的算法实现


## 4. Weka中的工具类


## 参考文献
