# Adaboost算法
## 算法简介
Adaboost是一种集成学习算法，是英文"Adaptive Boosting"（自适应增强）的缩写，AdaBoost方法的自适应在于：前一个分类器分错的样本会被用来训练下一个分类器。AdaBoost方法中使用的分类器可能很弱（比如出现很大错误率），但只要它的分类效果比随机好一点（比如两类问题分类错误率略小于0.5），就能够改善最终得到的模型。
AdaBoost方法是一种迭代算法，在每一轮中加入一个新的弱分类器，直到达到某个预定的足够小的错误率。每一个训练样本都被赋予一个权重，表明它被某个分类器选入训练集的概率。如果某个样本点已经被准确地分类，那么在构造下一个训练集中，它被选中的概率就被降低；相反，如果某个样本点没有被准确地分类，那么它的权重就得到提高。通过这样的方式，AdaBoost方法能“聚焦于”那些较难分（更富信息）的样本上。在具体实现上，最初令每个样本的权重都相等，对于第k次迭代操作，我们就根据这些权重来选取样本点，进而训练分类器Ck。然后就根据这个分类器，来提高被它分错的的样本的权重，并降低被正确分类的样本权重。然后，权重更新过的样本集被用于训练下一个分类器Ck。整个训练过程如此迭代地进行下去。(这一段摘自参考文献1，维基百科）
## 在Weka中使用
使用经典的iris数据集测试算法，我们给出其使用的样例。

```java
package src;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;

public class AdaboostTest {
	
	public void test_adaboost() {
		try {
			BufferedReader reader = new BufferedReader(
					new FileReader("/Applications/weka-3-7-12/data/iris.arff"));
			Instances data = new Instances(reader);
			reader.close();
			data.setClassIndex(data.numAttributes() - 1);
			
			// -S 1: 随机数种子；-I 10: 迭代次数, -W ...: 使用的弱分类器
			String opt_str = "-S 1 -I 10 -W weka.classifiers.trees.DecisionStump";
			String []options = weka.core.Utils.splitOptions(opt_str);
			AdaBoostM1 classifier = new AdaBoostM1();
			classifier.setOptions(options);
			classifier.buildClassifier(data);
			System.out.println(classifier);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	public static void main(String[] args) {
		AdaboostTest test = new AdaboostTest();
		test.test_adaboost();
		
	}

}

```
## 算法重点
直接来看`buildClassifier()`函数

```java
  public void buildClassifier(Instances data) throws Exception {

    // 初始化
    initializeClassifier(data);

    // Perform boosting iterations
    while (next()) {};

    // 完成收尾工作，具体实现是讲一些临时变量复制为NULL，等待垃圾回收
    done();
  }
```
这个函数结构写的很简单，先初始化，然后进行迭代地训练，最后再进行一些清理工作。所以，很显然，我们要关注的重点在next()函数中。


## 参考资料
[1] https://zh.wikipedia.org/zh/AdaBoost
[2] Weka源码