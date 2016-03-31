// Decompiled by Jad v1.5.8g. Copyright 2001 Pavel Kouznetsov.
// Jad home page: http://www.kpdus.com/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   myClassifier.java
package util;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class myClassifier {

    public myClassifier(Instances data, Classifier specific_classifier)
            throws Exception {
        my_classifier = Classifier.makeCopy(specific_classifier);
        my_classifier.buildClassifier(data);
    }

    public Classifier getClassifier() {
        return my_classifier;
    }


    public double TestAccuracy(Instances test)
            throws Exception {
        Evaluation eTest = new Evaluation(test);
        eTest.evaluateModel(Classifier.makeCopy(my_classifier), test);

        double correct = eTest.correct();
        double acc = (100D * correct) / (double) test.numInstances();
        return acc;
    }
    public double confidence(Instance in)
            throws Exception {
        double fDistribution[] = my_classifier.distributionForInstance(in);
        int indexOfMax = getMax_ofindex(fDistribution);
        return fDistribution[indexOfMax];
    }

    public double classifyInstance(Instance in)
            throws Exception {
        return my_classifier.classifyInstance(in);
    }

    private int getMax_ofindex(double d[]) {
        int max = 0;
        for (int i = 0; i < d.length; i++) {
            if (d[max] < d[i]) {
                max = i;
            }
        }

        return max;
    }

    private Classifier my_classifier;
}
