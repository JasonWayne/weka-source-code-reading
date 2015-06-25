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

    // boosting的过程
    while (next()) {};

    // 完成收尾工作，具体实现是讲一些临时变量复制为NULL，等待垃圾回收
    done();
  }
```
这个函数结构写的很简单，先初始化，然后进行迭代地训练，最后再进行一些清理工作。所以，很显然，我们要关注的重点在next()函数中。下面贴出next函数的代码，并且在添加注释进行解释。

```java
  public boolean next() throws Exception {

    // 是否达到参数中设定的迭代次数
    if (m_NumIterationsPerformed >= m_NumIterations) {
      return false;
    }

    // Debug时多输出信息
    if (m_Debug) {
      System.err.println("Training classifier "
                         + (m_NumIterationsPerformed + 1));
    }

    // 先进行一定比例的抽样
    Instances trainData = null;
    if (m_WeightThreshold < 100) {
      trainData = selectWeightQuantile(m_TrainingData,
                                       (double) m_WeightThreshold / 100);
    } else {
      trainData = new Instances(m_TrainingData);
    }

    double epsilon = 0;
    if ((m_UseResampling)
        || (!(m_Classifier instanceof WeightedInstancesHandler))) {


      int resamplingIterations = 0;
      double[] weights = new double[trainData.numInstances()];
      for (int i = 0; i < weights.length; i++) {
        weights[i] = trainData.instance(i).weight();
      }
      do {
        // 根据weights重新抽样
        Instances sample = trainData.resampleWithWeights(m_RandomInstance, weights);
        
       // 训练一个弱分类器
        m_Classifiers[m_NumIterationsPerformed].buildClassifier(sample);
        // 对弱分类器进行评价，这将影响到该分类器在最终的输出中的权重，以及下一次的抽样分布函数
        Evaluation evaluation = new Evaluation(m_TrainingData); 
        evaluation.evaluateModel(m_Classifiers[m_NumIterationsPerformed],
                                 m_TrainingData);
        epsilon = evaluation.errorRate();
        resamplingIterations++;
      } while (Utils.eq(epsilon, 0)
               && (resamplingIterations < MAX_NUM_RESAMPLING_ITERATIONS));
    } else {

      // 若不进行重采样，则随机
      if (m_Classifiers[m_NumIterationsPerformed] instanceof Randomizable) {
        ((Randomizable) m_Classifiers[m_NumIterationsPerformed])
          .setSeed(m_RandomInstance.nextInt());
      }
      m_Classifiers[m_NumIterationsPerformed].buildClassifier(trainData);

      // 同样需要评价，并且记下错误率
      Evaluation evaluation = new Evaluation(m_TrainingData); // Does this need to be a copy
      evaluation.evaluateModel(m_Classifiers[m_NumIterationsPerformed],
                               m_TrainingData);
      epsilon = evaluation.errorRate();
    }

    // 错误率大于0.5或为0时停止训练
    if (Utils.grOrEq(epsilon, 0.5) || Utils.eq(epsilon, 0)) {
      if (m_NumIterationsPerformed == 0) {
        m_NumIterationsPerformed = 1; // If we're the first we have to use it
      }
      return false;
    }

    // 根据这一次训练的弱分类器的错误率，重新设置样本抽样的概率分布
    double reweight = (1 - epsilon) / epsilon;
    // 以及在最后的分类器中的权重
    m_Betas[m_NumIterationsPerformed] = Math.log(reweight);
    if (m_Debug) {
      System.err.println("\terror rate = " + epsilon + "  beta = "
                         + m_Betas[m_NumIterationsPerformed]);
    }
    
    // 更新样本权重
    setWeights(m_TrainingData, reweight);

    // 完成这一轮的迭代
    m_NumIterationsPerformed++;
    return true;
  }

```
整个训练过程的逻辑还是非常清楚的，每一轮都会训练一个弱分类器，根据这个弱分类器的错分情况，调整下一次样本被采样的权重，这样让分类器关注于被错分的样本。同时，错误率还会决定每个分类器在最重的总分类器中的权重。最终的权重依据如下公式：$$\beta_k = \frac12ln\frac{1 - E_k}{E_k}$$

```java
    // 根据这一次训练的弱分类器的错误率，重新设置样本抽样的概率分布
    double reweight = (1 - epsilon) / epsilon;
    // 以及在最后的分类器中的权重
    m_Betas[m_NumIterationsPerformed] = Math.log(reweight);
```
我们再来看看总的分类器是如何去分类一个样本的。

```java
  public double[] distributionForInstance(Instance instance) throws Exception {

    // ...省略了一些条件判断
    
    // sums数组记录下每一个类，分类器的得分
    double[] sums = new double[instance.numClasses()];

    if (m_NumIterationsPerformed == 1) {
      return m_Classifiers[0].distributionForInstance(instance);
    } else {
      for (int i = 0; i < m_NumIterationsPerformed; i++) {
      // 关键看这一行代码
      // m_Betas数组是之前存下来的每一个分类器的错误率
        sums[(int) m_Classifiers[i].classifyInstance(instance)] += m_Betas[i];
      }
      // 转换成log概率输出
      return Utils.logs2probs(sums);
    }
  }
``` 
上述代码可以对应为如下公式：
$$ g(x)=\sum_{n=1}^{N}\beta_ih_i(x) $$, 
$$\beta_i$$对应于m\_Betas[i], N就是训练时的迭代次数，$$ h_i(x) $$就是每一个弱分类器，对应代码中的m\_Classifiers[i]。


## 参考资料
[1] https://zh.wikipedia.org/zh/AdaBoost

[2] Weka源码