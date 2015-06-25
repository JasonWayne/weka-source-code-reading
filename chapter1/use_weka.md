#使用Weka
我们总结了使用weka的四种方式，下面一一介绍。
##图形界面
这应该是最常用的方式，也是Weka流行的原因。
##命令行调用

```
java -cp WEKA_INS/weka.jar weka.classifiers.functions.Logistic 
-t WEKA_INS/data/weather.numeric.arff 
-T WEKA_INS/data/weather.numeric.arff 
-d ./weather.numeric.model.arff
```
其中，`WEKA_INS`为Weka的安装地址，`-t`用于设置训练集，`-T`设置测试集，这里分别用Weka自带的天气数据作为训练和测试集。


##java调用
##其他语言的接口调用

## 本节参考文献
[1] http://stats.stackexchange.com/questions/71684/how-to-interpret-weka-logistic-regression-output