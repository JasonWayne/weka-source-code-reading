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
