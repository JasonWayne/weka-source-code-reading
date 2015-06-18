# AbstractClassifer

Weka中的分类器都继承自此类，下面对这个类进行详细的剖析，先从导入的包讲起，这里只导入了三个Java的基础库。

```java
import java.io.Serializable;	// 用于对象的序列化
import java.util.Enumeration;	// 类似于Iterator，输出分类器的选项时用到
import java.util.Vector;		// 用Vector来保存选项

import weka.core.*;				// 导入了core下的所有包，用到时再说明
```

再看看这个类都继承了哪些接口。

```java
public abstract class AbstractClassifier implements Classifier, Cloneable,
  Serializable, OptionHandler, CapabilitiesHandler, RevisionHandler,
  CapabilitiesIgnorer {
 }
```
继承了很多个接口，下面挑选重要的说明。最重要的肯定是Classfier这个接口，来看看它的代码。

```java
// 所有的分类器都要继承此接口，
// 一个分类器至少要实现这里定义的classifyInstance()或distributionForInstance()二者之一
public interface Classifier ｛
	// 用于训练分类器，实现时要注意两点：
	// 1. 初始化没有被设定的参数；2. 训练时不能改变Instance的内容
	public abstract void buildClassifier(Instances data) throws Exception;
	
	public double classifyInstance(Instance instance) throws Exception;
	public double[] distributionForInstance(Instance instance) throws Exception;
	public Capabilities getCapabilities();
}
```