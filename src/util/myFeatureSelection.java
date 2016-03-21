// Decompiled by Jad v1.5.8g. Copyright 2001 Pavel Kouznetsov.
// Jad home page: http://www.kpdus.com/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   myFeatureSelection.java
package util;

import weka.attributeSelection.*;
import weka.classifiers.Classifier;

public class myFeatureSelection extends AttributeSelection {

    public myFeatureSelection(int featureNumToSelect, Classifier c) {
        NumToSelect = featureNumToSelect;
        setFeatureSelection();
    }

    private void setFeatureSelection(Classifier c) {
        WrapperSubsetEval my_evaluator = new WrapperSubsetEval();
        my_evaluator.setClassifier(c);
        my_evaluator.setFolds(2);
        setEvaluator(my_evaluator);
        GreedyStepwise my_search = new GreedyStepwise();
        my_search.setGenerateRanking(true);
        my_search.setNumToSelect(NumToSelect);
        setSearch(my_search);
    }

    private void setFeatureSelection() {
        GainRatioAttributeEval my_evaluator = new GainRatioAttributeEval();
        setEvaluator(my_evaluator);
        Ranker my_search = new Ranker();
        my_search.setNumToSelect(NumToSelect);
        setSearch(my_search);
    }

    private int NumToSelect;
}
