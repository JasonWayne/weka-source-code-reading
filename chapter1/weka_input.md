# Weka的输入
## .arff格式
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
## 其他格式
对于其他格式的文件，比如csv格式，Weka提供了方便的类进行转换，即将csv转化为arff格式后再进行使用，下面是Weka的官方中给出的转换的样例代码。

```java

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
 
import java.io.File;
 
public class CSV2Arff {
  /**
   * takes 2 arguments:
   * - CSV input file
   * - ARFF output file
   */
  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      System.out.println("\nUsage: CSV2Arff <input.csv> <output.arff>\n");
      System.exit(1);
    }
 
    // load CSV
    CSVLoader loader = new CSVLoader();
    loader.setSource(new File(args[0]));
    Instances data = loader.getDataSet();
 
    // save ARFF
    ArffSaver saver = new ArffSaver();
    saver.setInstances(data);
    saver.setFile(new File(args[1]));
    saver.setDestination(new File(args[1]));
    saver.writeBatch();
  }
}
```
除此之外，在weka的图形界面中，也可进行方便地转换。

## 参考资料
[1] https://weka.wikispaces.com/Converting+CSV+to+ARFF