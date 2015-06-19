# Evaluation类
Evaluation类顾名思义，用来评价分类器的性能。Weka中有两个Evaluation类，分别位于`weka/classifiers/evaluation/Evaluation.java`以及`weka/classifiers/Evaluation.java`，而且这两个类定义了同样的接口，其中`evaluation`包下的`Evaluation`类就是把所有的操作交给`classifier/Evalution.java`来完成，应该是一个历史遗留问题，可能在新版本中编写了单独的包`evalution`，因此把`Evalution.java`移动到了该包，而又为了能够适配旧版本已经编写的代码，就保留了classifier包下的`Evalution.java`。
Weka中，我们可以通过两种方式来评价一个分类器，下面分别介绍。

###方法1: 利用Evaluation类的evaluationModel，自行传入测试集。

```java
		try {
			//以经典的iris数据集作测试，weka自带该数据集
			BufferedReader reader = new BufferedReader(
					new FileReader("/Applications/weka-3-7-12/data/iris.arff"));
			Instances data = new Instances(reader);
			reader.close();
			data.setClassIndex(data.numAttributes() - 1);
			
			// 用data训练一个J48分类器
			J48 tree = new J48();
			tree.buildClassifier(data);
			
			// 这里传入的data是训练集的数据，用来获取一些信息，并不用来评价分类器
			Evaluation eval = new Evaluation(data);
			// 这里传入的是真正用来测试的data，为了简单，我们采用与训练集同样的数据作测试
			eval.evaluateModel(tree, data);
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
```
下面是输出信息：

```
Results
======

Correctly Classified Instances         147               98      %
Incorrectly Classified Instances         3                2      %
Kappa statistic                          0.97  
Mean absolute error                      0.0233
Root mean squared error                  0.108 
Relative absolute error                  5.2482 %
Root relative squared error             22.9089 %
Coverage of cases (0.95 level)          98.6667 %
Mean rel. region size (0.95 level)      34      %
Total Number of Instances              150     
```
### 方法2: 利用Evaluation类的crossValidationModel()方法，做k-fold cross-validation
对于k fold cross-validation(K折交叉验证)的概念，可以参考参考资料2种维基百科的解释，下面同样给出示例代码。在k fold cross-validation时，每一个数据都会有(k - 1)次选作训练集，1次选作测试集，也就是说，对于每一个样本，都会被测试一次，判断是否分类正确，因此仍然可以得到一个总的分类错误率。

```java
	public void test_cross_validation() {
		try {
			BufferedReader reader = new BufferedReader(
					new FileReader("/Applications/weka-3-7-12/data/iris.arff"));
			Instances data = new Instances(reader);
			reader.close();
			data.setClassIndex(data.numAttributes() - 1);
			
			J48 tree = new J48();
			tree.buildClassifier(data);
			
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(tree, data, 3, new Random());
			System.out.println("1");
			System.out.println(eval.toClassDetailsString()); 
			System.out.println("2");
			System.out.println(eval.toSummaryString()); 
			System.out.println("3");
			//输出混淆矩阵
			System.out.println(eval.toMatrixString());
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
```
输出：这次较前面一个例子，多输出了一些结果，这样就和WekaGUI中的分类器输出结果很相似了。

```
1
=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 0.980    0.000    1.000      0.980    0.990      0.985    0.990     0.987     Iris-setosa
                 0.940    0.030    0.940      0.940    0.940      0.910    0.967     0.920     Iris-versicolor
                 0.960    0.030    0.941      0.960    0.950      0.925    0.974     0.934     Iris-virginica
Weighted Avg.    0.960    0.020    0.960      0.960    0.960      0.940    0.977     0.947     

2

Correctly Classified Instances         144               96      %
Incorrectly Classified Instances         6                4      %
Kappa statistic                          0.94  
Mean absolute error                      0.0352
Root mean squared error                  0.1585
Relative absolute error                  7.9113 %
Root relative squared error             33.6102 %
Coverage of cases (0.95 level)          96.6667 %
Mean rel. region size (0.95 level)      34.2222 %
Total Number of Instances              150     

3
=== Confusion Matrix ===

  a  b  c   <-- classified as
 49  1  0 |  a = Iris-setosa
  0 47  3 |  b = Iris-versicolor
  0  2 48 |  c = Iris-virginica

```
## 参考资料
[1] Weka源码

[2] Cross-validation(statistics), https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation

[3] Weka开发［3］－Evaluation类, http://quweiprotoss.blog.163.com/blog/static/408828832008103042734622/