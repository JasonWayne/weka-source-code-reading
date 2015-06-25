#Logistic Regression
##算法简介
实质上来讲，Logistic Regression就是一个被logistic方程归一化后的线性回归。
在weka中最优化用的是拟牛顿方法。

##使用
该算法主要由两个参数：

* `maxIts`即最大迭代次数，默认值为-1，一直迭代到收敛
* `rigde`用来调整数组的模长对cost函数影响的大小，默认为1E-8

参数作为一个Options数组传递

训练并评价结果

```java
m_classifier.buildClassifier(newInstancesTrain);   
Evaluation eval = newEvaluation(newInstancesTrain); 
eval.evaluateModel(m_classifier, newInstancesTest);
```
##logistic算法流程
###计算L
$$
L = \\
-\sum_{i=1}^n\sum_{j=1}^{k-1}(Y_{ij} * log(P_j(X_i)))\\
+(1 - \sum_{j=1}^{k-1}Y_{ij})* log(1 - \sum_{j=1}^{k-1}P_j(X_i))\\
+ ridge * (B^2)
$$

```java
protected double objectiveFunction(double[] x) {
  double nll = 0; // -LogLikelihood
  int dim = m_NumPredictors + 1; // Number of variables per class

  for (int i = 0; i < cls.length; i++) { // ith instance

    double[] exp = new double[m_NumClasses - 1];
    int index;
    for (int offset = 0; offset < m_NumClasses - 1; offset++) {
      index = offset * dim;
      for (int j = 0; j < dim; j++) {
        exp[offset] += m_Data[i][j] * x[index + j];
      }
    }
    double max = exp[Utils.maxIndex(exp)];
    double denom = Math.exp(-max);
    double num;
    if (cls[i] == m_NumClasses - 1) { // Class of this instance
      num = -max;
    } else {
      num = exp[cls[i]] - max;
    }
    for (int offset = 0; offset < m_NumClasses - 1; offset++) {
      denom += Math.exp(exp[offset] - max);
    }

    nll -= weights[i] * (num - Math.log(denom)); // Weighted NLL
  }

  // Ridge: note that intercepts NOT included
  for (int offset = 0; offset < m_NumClasses - 1; offset++) {
    for (int r = 1; r < dim; r++) {
      nll += m_Ridge * x[offset * dim + r] * x[offset * dim + r];
    }
  }

    return nll;
}
```
###构建分类器
去除缺失值

去除无用值

将属性转化为0-1值

```java
 // Replace missing values
    m_ReplaceMissingValues = new ReplaceMissingValues();
    m_ReplaceMissingValues.setInputFormat(train);
    train = Filter.useFilter(train, m_ReplaceMissingValues);

    // Remove useless attributes
    m_AttFilter = new RemoveUseless();
    m_AttFilter.setInputFormat(train);
    train = Filter.useFilter(train, m_AttFilter);

    // Transform attributes
    m_NominalToBinary = new NominalToBinary();
    m_NominalToBinary.setInputFormat(train);
    train = Filter.useFilter(train, m_NominalToBinary);
```
###数据转换
```java
for (int i = 0; i < nC; i++) {
      // initialize X[][]
      Instance current = train.instance(i);
      Y[i] = (int) current.classValue(); // Class value starts from 0
      weights[i] = current.weight(); // Dealing with weights
      totWeights += weights[i];
    
      m_Data[i][0] = 1;
      int j = 1;
      for (int k = 0; k <= nR; k++) {
        if (k != m_ClassIndex) {
          double x = current.value(k);
          m_Data[i][j] = x;
          xMean[j] += weights[i] * x;
          xSD[j] += weights[i] * x * x;
          j++;
        }
      }
    
      // Class count
      sY[Y[i]]++;
    }
```
区别于一般的logistic回归算法，weka中的会计算每一条样本的权重。此段代码中对所有样本进行遍历，其中：

* Y[i]记录下每个样本的类别值,
* weight 记录下当前样本的权重,
* totWeights加和统计所有样本的权重和,
* m_Data 第二维是从 1 开始记录属性值的
* nR是属性数量
* nK是label数量

计算均值和标准差，使用z-score方法归一化。
在Optimization中使用拟牛顿法迭代，来寻找参数的估计值
###计算结果
采用优化后参数值来计算最后的分布

$$P_j(X_i)= \frac{1}{1+\sum_{i=1}^{k-1}e^{X_i*B_j}}$$

```java
private double[] evaluateProbability(double[] data) {
  double[] prob = new double[m_NumClasses], v = new double[m_NumClasses];

  // Log-posterior before normalizing
  for (int j = 0; j < m_NumClasses - 1; j++) {
    for (int k = 0; k <= m_NumPredictors; k++) {
      v[j] += m_Par[k][j] * data[k];
    }
  }
  v[m_NumClasses - 1] = 0;

  // Do so to avoid scaling problems
  for (int m = 0; m < m_NumClasses; m++) {
    double sum = 0;
    for (int n = 0; n < m_NumClasses - 1; n++) {
      sum += Math.exp(v[n] - v[m]);
    }
    prob[m] = 1 / (sum + Math.exp(-v[m]));
  }

  return prob;
}
```

##others
* weka对不需要的数组进行了内存释放，这是我平时不会想到的
`m_Data = null;`


##参考文献
https://en.wikipedia.org/wiki/Logistic_regression

