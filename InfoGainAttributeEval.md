#InfoGainAttributeEval
##算法描述

InfoGainAttributeEval是用一个单个属性评估器，通过计算类别对应属性的的信息增益来评估属性。
信息增益，按名称来理解的话，就是前后信息的差值，衡量一个属性区分以上数据样本的能力。常见于决策树中进行特征选择，weka中把它单独拿出，作为一种通用的特征选择方法。
##算法流程

```java

double[][][] counts = new double[data.numAttributes()][][]; //第一维，属性数量
    for (int k = 0; k < data.numAttributes(); k++) {
      if (k != classIndex) {
        int numValues = data.attribute(k).numValues(); 
        counts[k] = new double[numValues + 1][numClasses + 1]; //第二、三维
      }
    }

    double[] temp = new double[numClasses + 1];
    for (int k = 0; k < numInstances; k++) {
      Instance inst = data.instance(k);
      if (inst.classIsMissing()) {
        temp[numClasses] += inst.weight(); //样本权重
      } else {
        temp[(int) inst.classValue()] += inst.weight();
      }
    }
    for (int k = 0; k < counts.length; k++) {
      if (k != classIndex) {
        for (int i = 0; i < temp.length; i++) {
          counts[k][0][i] = temp[i];
        }
      }
    
```
使用3维数组count来储存各种数量值，第一维是属性个数，第二维是属性值的数量，第三维是类别的数量。从操作上来看，就是一个二维数组里嵌套了一个二维数组。用temp数组来储存权重，如果类缺失，则在最后一列进行累加，否则在类值列累加。最后将temp中计算所得的值存入counts中。

```java
    for (int k = 0; k < numInstances; k++) {
      Instance inst = data.instance(k);
      for (int i = 0; i < inst.numValues(); i++) {
        if (inst.index(i) != classIndex) {
          if (inst.isMissingSparse(i) || inst.classIsMissing()) {
            if (!inst.isMissingSparse(i)) {
              counts[inst.index(i)][(int) inst.valueSparse(i)][numClasses] += inst //count第三维上
                .weight();
              counts[inst.index(i)][0][numClasses] -= inst.weight();
            } else if (!inst.classIsMissing()) {
              counts[inst.index(i)][data.attribute(inst.index(i)).numValues()][(int) inst
                .classValue()] += inst.weight(); //count第二维上
              counts[inst.index(i)][0][(int) inst.classValue()] -= inst
                .weight();
            } else {
              counts[inst.index(i)][data.attribute(inst.index(i)).numValues()][numClasses] += inst
                .weight();  //二三维的最后
              counts[inst.index(i)][0][numClasses] -= inst.weight();
            }
          } else {
            counts[inst.index(i)][(int) inst.valueSparse(i)][(int) inst
              .classValue()] += inst.weight();
            counts[inst.index(i)][0][(int) inst.classValue()] -= inst.weight();
          }
        }
      }
    
```
如果值缺失，就在最后一列进行累加，例如如果class值缺失，counts[inst.index(i)][0][numClasses]这一列上累加。就在如果都不缺失的话，就将这个属性的这个属性值的类别值的元素加上它的权重。

```java
    if (m_missing_merge) {

      for (int k = 0; k < data.numAttributes(); k++) {
        if (k != classIndex) {
          int numValues = data.attribute(k).numValues();

          // 计算总和
          double[] rowSums = new double[numValues];
          double[] columnSums = new double[numClasses];
          double sum = 0;
          for (int i = 0; i < numValues; i++) {
            for (int j = 0; j < numClasses; j++) {
              rowSums[i] += counts[k][i][j];
              columnSums[j] += counts[k][i][j];
            }
            sum += rowSums[i];
          }

          if (Utils.gr(sum, 0)) { //return 0和sum间小的那个值
            double[][] additions = new double[numValues][numClasses];

            // 计算每一行那个需要加上去
            for (int i = 0; i < numValues; i++) {
              for (int j = 0; j < numClasses; j++) {
                additions[i][j] = (rowSums[i] / sum) * counts[k][numValues][j];
              }
            }

            //计算每一列
            for (int i = 0; i < numClasses; i++) {
              for (int j = 0; j < numValues; j++) {
                additions[j][i] += (columnSums[i] / sum)
                  * counts[k][j][numClasses];
              }
            }
             //生成新的矩阵，对counts赋值
            double[][] newTable = new double[numValues][numClasses];
            for (int i = 0; i < numValues; i++) {
              for (int j = 0; j < numClasses; j++) {
                newTable[i][j] = counts[k][i][j] + additions[i][j];
              }
            }
            counts[k] = newTable;
          }
        }
      }
    }
```
将前面累加的最后一列再分会到前面的数组中，最后更新count

```java
    m_InfoGains = new double[data.numAttributes()];
    for (int i = 0; i < data.numAttributes(); i++) {
      if (i != classIndex) {
        m_InfoGains[i] = (ContingencyTables.entropyOverColumns(counts[i]) - ContingencyTables
          .entropyConditionedOnRows(counts[i]));
      }
    }
  }
```
在这里计算了信息增益，用m_InfoGains数组储存，调用了entropyOverColumns和entropyConditionedOnRows两个函数
我们先来看一下信息增益的公式,以特征A对训练集D的信息增益$g(D|A)$为例
$$g(D|A) = H(D)-H(D|A)$$
其中$H(D)$是集合D的经验熵,$H(D|A)$是特征A给定条件下的信息经验熵，可以看出entropyOverColumns计算的就是$H(D)$，entropyConditionedOnRows计算的是$H(D|A)$
下面我们来看一下这两段程序

```java
  public static double entropyOverColumns(double[][] matrix){
    
    double returnValue = 0, sumForColumn, total = 0;

    for (int j = 0; j < matrix[0].length; j++){
      sumForColumn = 0;
      for (int i = 0; i < matrix.length; i++) {
	sumForColumn += matrix[i][j];
      }
      returnValue = returnValue - lnFunc(sumForColumn);
      total += sumForColumn; 
    }
    if (Utils.eq(total, 0)) {
      return 0;
    }
    return (returnValue + lnFunc(total)) / (total * log2);
  }
```
 
 其中有一个lnFunc返回的是
$$num*log(num)$$


```java
public static double lnFunc(double num){
    
    if (num <= 0) {
      return 0;
    } else {

      // Use cache if we have a sufficiently small integer
      if (num < MAX_INT_FOR_CACHE_PLUS_ONE) {
        int n = (int)num;
        if ((double)n == num) {
          return INT_N_LOG_N_CACHE[n];
        }
      }
      return num * Math.log(num);
    }
  }
```
 
 我们返回来再看entropyOverColumns，sumForColumn计算了matrix中所有项的和，也就是count[i]中的那个数组，由前可知，count[i]即count的第一维，储存的是各个
 returnValue是：
 $$-\sum_{i=0}^{M}C_i*logC_i$$
 最终返回的熵的表示形式为：
 $$\frac{-\sum_{i=0}^{M}C_i*logC_i+N*logN}{N*log2} $$
 
这与我们一般见到的熵的表达式不同，但是可以将下面的$N*log2$合并到上面，可以得到熵的一般表达形式。
$$-\sum_{i=0}^{M}p(i)*logp(i)$$

```java
  public static double entropyConditionedOnRows(double[][] matrix) {
    
    double returnValue = 0, sumForRow, total = 0;

    for (int i = 0; i < matrix.length; i++) {
      sumForRow = 0;
      for (int j = 0; j < matrix[0].length; j++) {
	returnValue = returnValue + lnFunc(matrix[i][j]);
	sumForRow += matrix[i][j];
      }
      returnValue = returnValue - lnFunc(sumForRow);
      total += sumForRow;
    }
    if (Utils.eq(total, 0)) {
      return 0;
    }
    return -returnValue / (total * log2);
  }

```
entropyConditionedOnRows计算了信息增益后半部分经验条件熵，与entropyOverColumns类似，sumForColumn计算了matrix中所有项的和

返回值的表达式为
$$-\frac{\sum_i^M(\sum_j^NA_j*logA_j-C_i*logC_i)}{N*log2}$$

##数据测试

我们使用Weka自带的劳工谈判数据集―labor.arff，该数据集是由那些在商界和个人服务领域之间，为至少有500人的组织达成的集体协议，这些组织中有教师、护士、大学老师、警察等等，结果是是否合同被认为可以接受，或者是不能接受。这个数据集有57个实例，17个属性，这个数据集存在很多残缺值，基本所有的属性都或多或少存在残缺值。

结合Naive Bayes算法，我们使用原本InfoGainAttributeEval算法进行特征选择与全部特征的分类结果进行对比。

这是使用全部17个特征的分类结果。

```
=== Summary ===

Correctly Classified Instances          51               89.4737 %
Incorrectly Classified Instances         6               10.5263 %
Kappa statistic                          0.7741
Mean absolute error                      0.1042
Root mean squared error                  0.2637
Relative absolute error                 22.7763 %
Root relative squared error             55.2266 %
Total Number of Instances               57     
```

使用InfoGainAttributeEval算法进行特征筛选，可以得到各个特征的信息增益值排名，选取前10个特征

```
Ranked attributes:
Search Method:
	Attribute ranking.

Attribute Evaluator (supervised, Class (nominal): 17 class):
	Information Gain Ranking Filter

Ranked attributes:
 0.2948   2 wage-increase-first-year
 0.1893   3 wage-increase-second-year
 0.1624  11 statutory-holidays
 0.1341  14 contribution-to-dental-plan
 0.1164  16 contribution-to-health-plan
 0.1091  12 vacation
 0.0855  13 longterm-disability-assistance
 0.0717   9 shift-differential
 0.0548   7 pension
 0.0484   5 cost-of-living-adjustment

Selected attributes: 2,3,11,14,16,12,13,9,7,5 : 10
```
这是特征筛选过的结果，可以看到看到分类正确的数量提高了一个……准确度从89.4737%提高到了91。2281%。

```
=== Summary ===

Correctly Classified Instances          52               91.2281 %
Incorrectly Classified Instances         5                8.7719 %
Kappa statistic                          0.8096
Mean absolute error                      0.1085
Root mean squared error                  0.2707
Relative absolute error                 23.7135 %
Root relative squared error             56.7045 %
Total Number of Instances               57     

```

总体来讲，在机器学习的过程中，进行特征的选择还是很有必要的，有些特征属于无用特征，加入之后反而会降低最后的效果，尤其是在低维度数量的特征中，每一个特征的权重都相对较高，如果像百度工程师所描述的几亿维数据，可能在之上进行特征筛选就不那么合适了。


##tips
* 在实际程序中，公式的计算可能会采取一些不同的方法，譬如此算法在计算熵时的没有直接计算概率，而是在最后整体除了一个$N*log2$,这样在计算时只有M次加法，一次除法，比直接计算概率要进行M次除法，时间效率要高的多。


##参考文献
统计学习方法 5.22 信息增益
