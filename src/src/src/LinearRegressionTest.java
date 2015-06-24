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
