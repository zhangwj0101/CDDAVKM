package util;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.WrapperSubsetEval;
import weka.classifiers.Classifier;

public class myFeatureSelection extends AttributeSelection {

    private int NumToSelect;

    public myFeatureSelection(int featureNumToSelect, Classifier c) {
        this.NumToSelect = featureNumToSelect;

        setFeatureSelection();
    }

    private void setFeatureSelection(Classifier c) {
        WrapperSubsetEval my_evaluator = new WrapperSubsetEval();
        my_evaluator.setClassifier(c);
        my_evaluator.setFolds(2);
        setEvaluator(my_evaluator);

        GreedyStepwise my_search = new GreedyStepwise();
        my_search.setGenerateRanking(true);
        my_search.setNumToSelect(this.NumToSelect);
        setSearch(my_search);
    }

    private void setFeatureSelection() {
        GainRatioAttributeEval my_evaluator = new GainRatioAttributeEval();
        setEvaluator(my_evaluator);

        Ranker my_search = new Ranker();
        my_search.setNumToSelect(this.NumToSelect);
        setSearch(my_search);
    }
}
