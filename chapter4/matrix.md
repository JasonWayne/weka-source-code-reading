# Matrix类
这个类是Weka中一个很重要的工具类，提供了很多矩阵的操作。理解这个类，对于看其他分类器的源码非常有帮助，同时，也是对于线性代数知识的很好复习。我们将对这个类中的函数进行详细的阅读。

```java

public class Matrix implements Cloneable, Serializable, RevisionHandler {

  // 用于序列化
  private static final long serialVersionUID = 7856794138418366180L;

  // 实际是保存成了二维数组，那么稀疏的矩阵怎么办?
  protected double[][] A;

  // m: 行， n: 列
  protected int m, n;
  
  // 初始化为全零矩阵
  public Matrix(int m, int n) {
    this.m = m;
    this.n = n;
    A = new double[m][n];
  }

  // 初始化为s
  public Matrix(int m, int n, double s) {
    this.m = m;
    this.n = n;
    A = new double[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        A[i][j] = s;
      }
    }
  }

  // 初始化为一个数组的值
  public Matrix(double[][] A) {
    m = A.length;
    n = A[0].length;
    for (int i = 0; i < m; i++) {
      if (A[i].length != n) {
        throw new IllegalArgumentException(
          "All rows must have the same length.");
      }
    }
    this.A = A;
  }

  // 初始化为一个数组的值
  public Matrix(double[][] A, int m, int n) {
  	...
  }

  // 深拷贝，会生成一个新的Matrix对象
  public Matrix copy() {
    Matrix X = new Matrix(m, n);
    double[][] C = X.getArray();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        C[i][j] = A[i][j];
      }
    }
    return X;
  }

  // 同copy()
  @Override
  public Object clone() {
    return this.copy();
  }
  
  // 下面是很多get, set方法，在此略过


  // 判断是否为对称阵
  public boolean isSymmetric() {
    int nr = A.length, nc = A[0].length;
    if (nr != nc) {
      return false;
    }

    for (int i = 0; i < nc; i++) {
      for (int j = 0; j < i; j++) {
        if (A[i][j] != A[j][i]) {
          return false;
        }
      }
    }
    return true;
  }

  // 判断是否为方阵
  public boolean isSquare() {
    return (getRowDimension() == getColumnDimension());
  }

  // 转置
  public Matrix transpose() {
    Matrix X = new Matrix(n, m);
    double[][] C = X.getArray();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        C[j][i] = A[i][j];
      }
    }
    return X;
  }

  // 返回和最大的列的和的值
  public double norm1() {
    double f = 0;
    for (int j = 0; j < n; j++) {
      double s = 0;
      for (int i = 0; i < m; i++) {
        s += Math.abs(A[i][j]);
      }
      f = Math.max(f, s);
    }
    return f;
  }

  // 返回最大的奇艺值，委托给了奇异值分解类来计算
  public double norm2() {
    return (new SingularValueDecomposition(this).norm2());
  }

  // 返回和最大的行的和的值
  public double normInf() {
    double f = 0;
    for (int i = 0; i < m; i++) {
      double s = 0;
      for (int j = 0; j < n; j++) {
        s += Math.abs(A[i][j]);
      }
      f = Math.max(f, s);
    }
    return f;
  }

  // 所有元素取相反数
  public Matrix uminus() {
    Matrix X = new Matrix(m, n);
    double[][] C = X.getArray();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        C[i][j] = -A[i][j];
      }
    }
    return X;
  }

  // 矩阵求和
  public Matrix plus(Matrix B) {
    // 先检验
    checkMatrixDimensions(B);
    Matrix X = new Matrix(m, n);
    double[][] C = X.getArray();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        C[i][j] = A[i][j] + B.A[i][j];
      }
    }
    return X;
  }

  // 求和时值加在当前矩阵上
  public Matrix plusEquals(Matrix B) {
    checkMatrixDimensions(B);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        A[i][j] = A[i][j] + B.A[i][j];
      }
    }
    return this;
  }

  // 减法
  public Matrix minus(Matrix B) {
    checkMatrixDimensions(B);
    Matrix X = new Matrix(m, n);
    double[][] C = X.getArray();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        C[i][j] = A[i][j] - B.A[i][j];
      }
    }
    return X;
  }

  // 减在当前矩阵上
  public Matrix minusEquals(Matrix B) {
    checkMatrixDimensions(B);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        A[i][j] = A[i][j] - B.A[i][j];
      }
    }
    return this;
  }

  // element wise mulplication， 即 .*
  public Matrix arrayTimes(Matrix B) {
    checkMatrixDimensions(B);
    Matrix X = new Matrix(m, n);
    double[][] C = X.getArray();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        C[i][j] = A[i][j] * B.A[i][j];
      }
    }
    return X;
  }

  // 同上，但是是in-place, 即值保留在当前矩阵
  public Matrix arrayTimesEquals(Matrix B) {
    checkMatrixDimensions(B);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        A[i][j] = A[i][j] * B.A[i][j];
      }
    }
    return this;
  }

  // ./
  public Matrix arrayRightDivide(Matrix B) {
    checkMatrixDimensions(B);
    Matrix X = new Matrix(m, n);
    double[][] C = X.getArray();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        C[i][j] = A[i][j] / B.A[i][j];
      }
    }
    return X;
  }

  // ./, inplace
  public Matrix arrayRightDivideEquals(Matrix B) {
    checkMatrixDimensions(B);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        A[i][j] = A[i][j] / B.A[i][j];
      }
    }
    return this;
  }

  // B除以A
  public Matrix arrayLeftDivide(Matrix B) {
    checkMatrixDimensions(B);
    Matrix X = new Matrix(m, n);
    double[][] C = X.getArray();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        C[i][j] = B.A[i][j] / A[i][j];
      }
    }
    return X;
  }

  // B除以A，inplace
  public Matrix arrayLeftDivideEquals(Matrix B) {
    checkMatrixDimensions(B);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        A[i][j] = B.A[i][j] / A[i][j];
      }
    }
    return this;
  }

  /**
   * Multiply a matrix by a scalar, C = s*A
   * 
   * @param s scalar
   * @return s*A
   */
  public Matrix times(double s) {
    Matrix X = new Matrix(m, n);
    double[][] C = X.getArray();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        C[i][j] = s * A[i][j];
      }
    }
    return X;
  }

  // 乘以一个系数，即在每个元素上都乘以该系数
  public Matrix timesEquals(double s) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        A[i][j] = s * A[i][j];
      }
    }
    return this;
  }

  // 矩阵乘法，并没有作优化
  public Matrix times(Matrix B) {
    if (B.m != n) {
      throw new IllegalArgumentException("Matrix inner dimensions must agree.");
    }
    Matrix X = new Matrix(m, B.n);
    double[][] C = X.getArray();
    double[] Bcolj = new double[n];
    for (int j = 0; j < B.n; j++) {
      for (int k = 0; k < n; k++) {
        Bcolj[k] = B.A[k][j];
      }
      for (int i = 0; i < m; i++) {
        double[] Arowi = A[i];
        double s = 0;
        for (int k = 0; k < n; k++) {
          // 逐个将A的第i行第k个元素和B的第j列第k个元素相乘相加
          s += Arowi[k] * Bcolj[k];
        }
        C[i][j] = s;
      }
    }
    return X;
  }

  // 将矩阵分解为一个上三角阵（L)和一个下三角阵(U)的乘积形式
  public LUDecomposition lu() {
    return new LUDecomposition(this);
  }

  // 把矩阵分解成一个半正交矩阵与一个上三角矩阵的积。QR分解经常用来解线性最小二乘法问题
  public QRDecomposition qr() {
    return new QRDecomposition(this);
  }

  // 奇异值分解，也有专门的类来处理
  public SingularValueDecomposition svd() {
    return new SingularValueDecomposition(this);
  }

  // 特征分解，将矩阵分解为由其特征值和特征向量表示的矩阵之积
  public EigenvalueDecomposition eig() {
    return new EigenvalueDecomposition(this);
  }

  // 解方程： A * X = B
  public Matrix solve(Matrix B) {
    return (m == n ? (new LUDecomposition(this)).solve(B)
      : (new QRDecomposition(this)).solve(B));
  }

  // 解方程 A' * X = B', 即
  public Matrix solveTranspose(Matrix B) {
    return transpose().solve(B.transpose());
  }

  // 矩阵求逆，实现方法是求A * X = I
  public Matrix inverse() {
    return solve(identity(m, m));
  }


  // 岭回归，X作为自变量，y作为因变量，ridge为岭回归系数
  public LinearRegression regression(Matrix y, double ridge) {
    return new LinearRegression(this, y, ridge);
  }
  
  // 求行列式的值，先做LU分解后求对角线元素的积
  public double det() {
    return new LUDecomposition(this).det();
  }

  // 求矩阵的秩, 委托给奇异值分解完成
  public int rank() {
    return new SingularValueDecomposition(this).rank();
  }


  // 矩阵的迹，即对角线元素之和
  public double trace() {
    double t = 0;
    for (int i = 0; i < Math.min(m, n); i++) {
      t += A[i][i];
    }
    return t;
  }

  // 生成一个填满随机数的矩阵，提供这样的方法会很方便。
  public static Matrix random(int m, int n) {
    Matrix A = new Matrix(m, n);
    double[][] X = A.getArray();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        X[i][j] = Math.random();
      }
    }
    return A;
  }

  // 生成单位矩阵，注意这里不一定是方阵
  public static Matrix identity(int m, int n) {
    Matrix A = new Matrix(m, n);
    double[][] X = A.getArray();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        // 这样写避免了if else的条件判断
        X[i][j] = (i == j ? 1.0 : 0.0);
      }
    }
    return A;
  }

  // 打印矩阵
  public void print(int w, int d) {
    print(new PrintWriter(System.out, true), w, d);
  }

]

  // 后面是很多读写，输入输出，与matlab格式转换的io函数，在此略过

  // Weka的几乎每个类都提供了一个简单的测试方法，这是一个很好的编程习惯
  public static void main(String[] args) {
    Matrix I;
    Matrix A;
    Matrix B;

    try {
      // 单位矩阵
      System.out.println("\nIdentity\n");
      I = Matrix.identity(3, 5);
      System.out.println("I(3,5)\n" + I);

      // 矩阵基本操作
      System.out.println("\nbasic operations - square\n");
      A = Matrix.random(3, 3);
      B = Matrix.random(3, 3);
      System.out.println("A\n" + A);
      System.out.println("B\n" + B);
      System.out.println("A'\n" + A.inverse());
      System.out.println("A^T\n" + A.transpose());
      System.out.println("A+B\n" + A.plus(B));
      System.out.println("A*B\n" + A.times(B));
      System.out.println("X from A*X=B\n" + A.solve(B));

      // basic operations - non square
      System.out.println("\nbasic operations - non square\n");
      A = Matrix.random(2, 3);
      B = Matrix.random(3, 4);
      System.out.println("A\n" + A);
      System.out.println("B\n" + B);
      System.out.println("A*B\n" + A.times(B));


      // 特征值分解
      System.out.println("\nEigenvalue Decomposition\n");
      EigenvalueDecomposition evd = A.eig();
      System.out.println("[V,D] = eig(A)");
      System.out.println("- V\n" + evd.getV());
      System.out.println("- D\n" + evd.getD());

      // LU 分解
      System.out.println("\nLU Decomposition\n");
      LUDecomposition lud = A.lu();
      System.out.println("[L,U,P] = lu(A)");
      System.out.println("- L\n" + lud.getL());
      System.out.println("- U\n" + lud.getU());
      System.out.println("- P\n" + Utils.arrayToString(lud.getPivot()) + "\n");

      // 回归
      System.out.println("\nRegression\n");
      B = new Matrix(new double[][] { { 3 }, { 2 } });
      double ridge = 0.5;
      double[] weights = new double[] { 0.3, 0.7 };
      System.out.println("A\n" + A);
      System.out.println("B\n" + B);
      System.out.println("ridge = " + ridge + "\n");
      System.out.println("weights = " + Utils.arrayToString(weights) + "\n");
      System.out.println("A.regression(B, ridge)\n" + A.regression(B, ridge)
        + "\n");
      System.out.println("A.regression(B, weights, ridge)\n"
        + A.regression(B, weights, ridge) + "\n");

      // 读写
      System.out.println("\nWriter/Reader\n");
      StringWriter writer = new StringWriter();
      A.write(writer);
      System.out.println("A.write(Writer)\n" + writer);
      A = new Matrix(new StringReader(writer.toString()));
      System.out.println("A = new Matrix.read(Reader)\n" + A);

      // Matlab格式转化
      System.out.println("\nMatlab-Format\n");
      String matlab = "[ 1   2;3 4 ]";
      System.out.println("Matlab: " + matlab);
      System.out.println("from Matlab:\n" + Matrix.parseMatlab(matlab));
      System.out
        .println("to Matlab:\n" + Matrix.parseMatlab(matlab).toMatlab());
      matlab = "[1 2 3 4;3 4 5 6;7 8 9 10]";
      System.out.println("Matlab: " + matlab);
      System.out.println("from Matlab:\n" + Matrix.parseMatlab(matlab));
      System.out.println("to Matlab:\n" + Matrix.parseMatlab(matlab).toMatlab()
        + "\n");
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}

```

## 收获
这样一遍通读下来整个Matrix的源码，主要有以下的收获：
 
- 编写方便的条件检查函数，比如在这里的checkMatrixDimensions()
- 对于矩阵的常用操作了解了其编程的实现，并且补充了很多遗忘的概念
- 将一些方法委托给其他类完成
- 为每一个类编写main函数，作为测试用例
- 熟悉了一个工具类的结构，以后自己编写工具类时会更顺手

## 参考资料
[1] https://en.wikipedia.org/wiki/LU_decomposition

[2] https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix

[3] https://en.wikipedia.org/wiki/QR_decomposition

[4] https://en.wikipedia.org/wiki/Singular_value_decomposition