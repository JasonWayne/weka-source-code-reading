package src;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;



public class Test {
	public void test() {
		try {
			BufferedReader reader = new BufferedReader(
					new FileReader("/Applications/weka-3-7-12/data/iris.arff"));
			Instances data = new Instances(reader);
			reader.close();
			data.setClassIndex(data.numAttributes() - 1);
			
//			String[] options = weka.core.Utils.splitOptions("-R 1");
			J48 tree = new J48();
//			tree.setOptions(options);
			tree.buildClassifier(data);
			
			Evaluation eval = new Evaluation(data);
			eval.evaluateModel(tree, data);
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {
		Test t = new Test();
		t.test();
	}
}
