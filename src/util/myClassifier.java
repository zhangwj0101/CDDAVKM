package util;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class myClassifier {

    private Classifier my_classifier;

    public myClassifier(Instances data, Classifier specific_classifier)
            throws Exception {
        this.my_classifier = Classifier.makeCopy(specific_classifier);
        this.my_classifier.buildClassifier(data);
    }

    public Classifier getClassifier() {
        return this.my_classifier;
    }

    public double TestAccuracy(Instances test) throws Exception {
        Evaluation eTest = new Evaluation(test);
        eTest.evaluateModel(Classifier.makeCopy(this.my_classifier), test);
        double correct = eTest.correct();
        double acc = 100.0D * correct / test.numInstances();
        return acc;
    }

    public double confidence(Instance in) throws Exception {
        double[] fDistribution = this.my_classifier.distributionForInstance(in);
        int indexOfMax = getMax_ofindex(fDistribution);
        return fDistribution[indexOfMax];
    }

    public double classifyInstance(Instance in) throws Exception {
        return this.my_classifier.classifyInstance(in);
    }

    private int getMax_ofindex(double[] d) {
        int max = 0;
        for (int i = 0; i < d.length; i++) {
            if (d[max] < d[i]) {
                max = i;
            }
        }
        return max;
    }
}
