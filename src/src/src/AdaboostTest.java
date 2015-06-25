package src;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;

public class AdaboostTest {
	
	public void test_adaboost() {
		try {
			BufferedReader reader = new BufferedReader(
					new FileReader("/Applications/weka-3-7-12/data/iris.arff"));
			Instances data = new Instances(reader);
			reader.close();
			data.setClassIndex(data.numAttributes() - 1);
			
			String opt_str = "-P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump";
			String []options = weka.core.Utils.splitOptions(opt_str);
			AdaBoostM1 classifier = new AdaBoostM1();
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
		AdaboostTest test = new AdaboostTest();
		test.test_adaboost();
		
	}

}
