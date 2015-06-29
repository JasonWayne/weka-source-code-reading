#Naive Bayes
##算法概述
对于给出的待分类项，求解在此项出现的条件下各个类别出现的概率，哪个最大，就认为此待分类项属于哪个类别。发源于古典数学理论，有着坚实的数学基础，以及稳定的分类效率。同时，NBC模型所需估计的参数很少，对缺失数据不太敏感，算法也比较简单。理论上，NBC模型与其他分类方法相比具有最小的误差率。但是实际上并非总是如此，这是因为NBC模型假设属性之间相互独立，这个假设在实际应用中往往是不成立的，这给NBC模型的正确分类带来了一定影响。
##算法流程
Naive Bayes的一般计算方法如下：

1. 设$x={a_1,a_2,...,a_m}$为一个待分类样本，而每个a为x的一个特征属性。
2. 类别标签集合$C={y_1,y_2,...,y_n}$
3. 计算$P(y_1|x),P(y_2|x),...,P(y_n|x),$
4. 如果$P(y_k|x)=max{P(y_i|x)}$,则$x\in{y_k}$

在计算条件概率时，根据bayes定理，有

$$P(y_i|x)=\frac{P(x|y_i)*P(y_i)}{P(x)}$$

因为分母对于所有类别为常数，各特征之间条件独立，所以我们可以得到

$$P(x|y_i)P(y_i)=P(a_1|y_i)P(a_2|y_i)...P(a_m|y_i)P(y_i)=P(y_i)\prod^m_{j=1}P(a_j|y_i)$$
##程序流程
在计算了一个`numPrecision`,这个值是在同一个属性上的数值不同，则相减得到值再都相加起来得到`deltaSum`，这样在最后将`deltaSum`除以数值不同的数量即`distinct` 得到`numPrecision`。在后面用作构建参数。

```java
public void buildClassifier(Instances instances) throws Exception {
……
if ((m_Instances.numInstances() > 0)
        && !m_Instances.instance(0).isMissing(attribute)) {
  double lastVal = m_Instances.instance(0).value(attribute);
  double currentVal, deltaSum = 0;
  int distinct = 0;
  for (int i = 1; i < m_Instances.numInstances(); i++) {
    Instance currentInst = m_Instances.instance(i);
    if (currentInst.isMissing(attribute)) {
      break;
    }
    currentVal = currentInst.value(attribute);
    if (currentVal != lastVal) {
      deltaSum += currentVal - lastVal;
      lastVal = currentVal;
      distinct++;
    }
  }
  if (distinct > 0) {
    numPrecision = deltaSum / distinct;
  }
}
……
}
```
`distributionForInstance`函数进行的就是我们所知道的Naive Bayes的过程了。首先计算每一个分类的概率$P(Y)$即程序中的`probs`数组，temp为$P(x|y_i)$，并将`prob`更新为$P(y)*\prod{P(a|y_i)}$,之后取最大的来更新max值。

```java

    double[] probs = new double[m_NumClasses];
    for (int j = 0; j < m_NumClasses; j++) {
      probs[j] = m_ClassDistribution.getProbability(j);
    }
    Enumeration<Attribute> enumAtts = instance.enumerateAttributes();
    int attIndex = 0;
    while (enumAtts.hasMoreElements()) {
      Attribute attribute = enumAtts.nextElement();
      if (!instance.isMissing(attribute)) {
        double temp, max = 0;
        for (int j = 0; j < m_NumClasses; j++) {
          temp = Math.max(1e-75, Math.pow(m_Distributions[attIndex][j]
            .getProbability(instance.value(attribute)),
            m_Instances.attribute(attIndex).weight()));
          probs[j] *= temp;
          if (probs[j] > max) {
            max = probs[j];
          }

```

在计算概率时，离散值可以直接处理，而对于连续值，weka假设其符合正态分布函数

* 首先计算p,需要先计算其均值和标准差，之后可根据公式计算

	$$p = \frac{data- \mu}{\sigma}$$
	
* 然后计算其正态分布函数
	
	$$namal(p) = \frac{1}{\sqrt{2\pi}}\int_{-inf}^pe^\frac{-t^2}{2}dt$$


```java
  public double getProbability(double data) {

    data = round(data);
    double zLower = (data - m_Mean - (m_Precision / 2)) / m_StandardDev;
    double zUpper = (data - m_Mean + (m_Precision / 2)) / m_StandardDev;

    double pLower = Statistics.normalProbability(zLower);
    double pUpper = Statistics.normalProbability(zUpper);
    return pUpper - pLower;
  }
  
  public static double normalProbability(double a) {

    double x, y, z;

    x = a * SQRTH;
    z = Math.abs(x);

    if (z < SQRTH) {
      y = 0.5 + 0.5 * errorFunction(x);
    } else {
      y = 0.5 * errorFunctionComplemented(z);
      if (x > 0) {
        y = 1.0 - y;
      }
    }
    return y;
  }
```

##tips
* 在处理数据时，并不是在原始数据上进行处理，而是复制一份新的数据，在程序中使用
* 总体来看`numPrecision`的作用在于在计算概率时，可以避免0值的出现
