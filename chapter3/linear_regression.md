# Linear Regression
## 算法介绍
在很多的机器学习课程（比如Andrew Ng在Coursera上的Machine Learning公开课）中，都以它作为讲解的第一个算法，它是很多算法（比如逻辑斯蒂回归，神经网络算法）的基础。

在Weka中，有两个相应的实现，SimpleLinearRegression类实现了单变量的线性回归，LinearRegression则是多变量的线性回归的实现，SimpleLinearRegression过于简单，并没有什么实际价值，因此，我们将重点关注LinearRegression。

## 在Weka中使用
我们用Weka跑一下经典的house数据集

```java
package src;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;

public class LinearRegressionTest {

	public void test_linear_regression() {
		try {
			BufferedReader reader = new BufferedReader(
					new FileReader("/Applications/weka-3-7-12/data/house.arff"));
			Instances data = new Instances(reader);
			reader.close();
			data.setClassIndex(data.numAttributes() - 1);
			
			//-S表示特征选择的方式, -R表示岭回归的系数
			String opt_str = "-S 0 -R 1.0E-8";
			String []options = weka.core.Utils.splitOptions(opt_str);
			LinearRegression classifier = new LinearRegression();
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
		LinearRegressionTest test = new LinearRegressionTest();
		test.test_linear_regression();
		
	}
}

```

下面是输出：

```
Linear Regression Model

sellingPrice =

    -26.6882 * houseSize +
      7.0551 * lotSize +
  43166.0767 * bedrooms +
  42292.0901 * bathroom +
 -21661.1208
 
```


## 算法重点
可以看到，前面对算法传入参数时，传入了两个参数，分别是-S和-R，因此，这也是在看算法源代码时所要关注的重点。先看buildClassfier()方法的代码，模型的参数是在这个方法中被训练的，由于这是第一个模型，我们将逐行来看源代码。

```java
  public void buildClassifier(Instances data) throws Exception {
  	//模型是否已经训练好
    m_ModelBuilt = false;

	
    if (!m_checksTurnedOff) {
      // 测试该模型是否适合该数据
      getCapabilities().testWithFail(data);

      if (m_outputAdditionalStats) {
		// 本模型不支持weights，这里需要进行测试
        boolean ok = true;
        for (int i = 0; i < data.numInstances(); i++) {
          if (data.instance(i).weight() != 1) {
            ok = false;
            break;
          }
        }
        if (!ok) {
          throw new Exception(
            "Can only compute additional statistics on unweighted data");
        }
      }      
     
      data = new Instances(data);
      // 删除空数据
      data.deleteWithMissingClass();

	   // 其他处理
      m_TransformFilter = new NominalToBinary();
      m_TransformFilter.setInputFormat(data);
      data = Filter.useFilter(data, m_TransformFilter);
      m_MissingFilter = new ReplaceMissingValues();
      m_MissingFilter.setInputFormat(data);
      data = Filter.useFilter(data, m_MissingFilter);
      data.deleteWithMissingClass();
    } else {
      m_TransformFilter = null;
      m_MissingFilter = null;
    }
```
这前面的一段都是在进行些判断和数据的清洗工作，下面则将是一些系数的初始化。

```java
	m_ClassIndex = data.classIndex();
    m_TransformedData = data;

    m_Coefficients = null;
	
	// 这个数组用于表示对应的变量是否会出现在最重的回归方程中
    m_SelectedAttributes = new boolean[data.numAttributes()];
    m_Means = new double[data.numAttributes()];
    m_StdDevs = new double[data.numAttributes()];
    for (int j = 0; j < data.numAttributes(); j++) {
      if (j != m_ClassIndex) {
        m_SelectedAttributes[j] = true; // Turn attributes on for a start
        m_Means[j] = data.meanOrMode(j);
        m_StdDevs[j] = Math.sqrt(data.variance(j));
        if (m_StdDevs[j] == 0) {
          m_SelectedAttributes[j] = false;
        }
      }
    }

    m_ClassStdDev = Math.sqrt(data.variance(m_TransformedData.classIndex()));
    m_ClassMean = data.meanOrMode(m_TransformedData.classIndex());
```
而真正的训练则是调用了findBestModel()方法:

```java
    findBestModel();
```
而在这个方法中，调用了doRegression方法来训练参数，并保存到m_Coefficients变量中:

```java
    do {
      m_Coefficients = doRegression(m_SelectedAttributes);
    } while (m_EliminateColinearAttributes
      && deselectColinearAttributes(m_SelectedAttributes, m_Coefficients));
```

在doRegression方法中，同样要进行一些预处理，我们在这里略过，直接找到训练参数的关键代码：

```java
	...
	// 自变量
	independent = new Matrix(m_TransformedData.numInstances(), numAttributes);
	// 因变量
	dependent = new Matrix(m_TransformedData.numInstances(), 1);
	...
	double[] coeffsWithoutIntercept =
        independent.regression(dependent, m_Ridge).getCoefficients();
      System.arraycopy(coeffsWithoutIntercept, 0, coefficients, 0,
        numAttributes);
```
可以看到，weka又是调用了Matrix类的regression方法去计算系数，那我们再来看这个方法.

```java
  public LinearRegression regression(Matrix y, double ridge) {
    return new LinearRegression(this, y, ridge);
  }
```
这里又new了一个LinearRegression类，这个LinearRegression类位于`/core/matrix`包下，在其构造函数中，用`calculate()`函数计算了模型的系数：

```java
  protected void calculate(Matrix a, Matrix y, double ridge) {

	// 检查
    if (y.getColumnDimension() > 1)
      throw new IllegalArgumentException("Only one dependent variable allowed");

	// 初始化
    int nc = a.getColumnDimension();
    m_Coefficients = new double[nc];
    Matrix solution;

    Matrix ss = aTa(a);
    Matrix bb = aTy(a, y);

    boolean success = true;

	// 真正的计算在这里
    do {
      Matrix ssWithRidge = ss.copy();
      for (int i = 0; i < nc; i++)
        ssWithRidge.set(i, i, ssWithRidge.get(i, i) + ridge);

      try {
        solution = ssWithRidge.solve(bb);
        for (int i = 0; i < nc; i++)
          m_Coefficients[i] = solution.get(i, 0);
        success = true;
      } catch (Exception ex) {
        ridge *= 10;
        success = false;
      }
    } while (!success);
  }
```
这里依然调用了Matrix类的`solve`方法，这个方法是用来解`A * X = B`这样的矩阵方程的，把这里的几个串起来，可以发现，Weka中使用了这个公式来计算系数的。
$$ \beta(k) = ( X' * X + kI) ^ {-1} * X * y $$
至此，我们跟踪到了Weka对于线性回归的系数的计算过程。这里岭回归的系数k相当于我们在其他机器学习方法中的正则化项。


## 本文中涉及的Weka类
`SimpleLinearRegression`, `LinearRegression`, `Matrix`, `/core/Matrix/LinearRegression`

## 参考文献
[1] http://en.wikipedia.org/wiki/Simple_linear_regression
[2] https://en.wikipedia.org/wiki/Linear_regression
[3] Weka 源码