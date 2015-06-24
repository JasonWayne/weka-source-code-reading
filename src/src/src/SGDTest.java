package src;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.functions.SGD;
import weka.core.Instances;

public class SGDTest {

	public void test_sgd() {
		try {
			BufferedReader reader = new BufferedReader(
					new FileReader("/Applications/weka-3-7-12/data/breast-cancer.arff"));
			Instances data = new Instances(reader);
			reader.close();
			data.setClassIndex(data.numAttributes() - 1);
			
			String opt_str = "-F 1 -L 0.01 -R 1.0E-4 -E 500 -C 0.001 -S 1";
			String []options = weka.core.Utils.splitOptions(opt_str);
			SGD classifier = new SGD();
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
		SGDTest test = new SGDTest();
		test.test_sgd();
		
	}

}
