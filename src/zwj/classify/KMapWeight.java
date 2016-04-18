// Decompiled by Jad v1.5.8g. Copyright 2001 Pavel Kouznetsov.
// Jad home page: http://www.kpdus.com/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   KMapWeight.java
package zwj.classify;

import zwj.dimesionReduction.KDA;

import java.io.*;
import java.util.Random;
import java.util.Scanner;

import zwj.util.myClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class KMapWeight {

    public KMapWeight(Instances In, Instances Out, int it, double ratio)
            throws Exception {
        Instances tmp[] = getLU(In, (int) ((double) In.numInstances() * ratio));
        InDomainU = new Instances(tmp[1]);
        InDomainL = new Instances(tmp[0]);
        OutDomain = new Instances(Out);
        All = new Instances(InDomainL);
        for (int i = 0; i < Out.numInstances(); i++) {
            All.add((Instance) Out.instance(i).copy());
        }

        maxIt = it;
        accuracy = new double[maxIt];
        betaVal = new double[maxIt];
        weightL = new double[InDomainL.numInstances()];
        weightO = new double[OutDomain.numInstances()];
        for (int i = 0; i < InDomainL.numInstances(); i++) {
            weightL[i] = 1.0D;
        }

        for (int i = 0; i < OutDomain.numInstances(); i++) {
            weightO[i] = 1.0D;
        }

    }

    private Instances selectL(double newIn[], int curIt)
            throws Exception {
        Instances ans = new Instances(InDomainL);
        double sumW = 0.0D;
        for (int i = 0; i < newIn.length; i++) {
            sumW += weightL[i];
        }

        double err = 0.0D;
        for (int i = 0; i < newIn.length; i++) {
            err += weightL[i] * Math.abs(newIn[i] - InDomainL.instance(i).classValue());
        }

        err /= sumW;
        double betaT = err / (1.0D - err);
        if (betaT == 0.0D) {
            betaT = 1.0000000000000001E-005D;
        }
        betaVal[curIt] = betaT;
        for (int i = 0; i < newIn.length; i++) {
            weightL[i] = weightL[i] * Math.pow(betaT, -Math.abs(newIn[i] - InDomainL.instance(i).classValue()));
            ans.instance(i).setWeight(weightL[i]);
        }

        Random rand = new Random((long) (1000000D * Math.random()));
        ans = ans.resampleWithWeights(rand);
        return ans;
    }

    private Instances selectO(double newOut[], int curIt)
            throws Exception {
        Instances ans = new Instances(OutDomain);
        double beta = 1.0D / (1.0D + Math.sqrt(2D * Math.log((double) newOut.length / (double) maxIt)));
        for (int i = 0; i < newOut.length; i++) {
            weightO[i] *= Math.pow(beta, Math.abs(newOut[i] - OutDomain.instance(i).classValue()));
            ans.instance(i).setWeight(weightO[i]);
        }

        Random rand = new Random((long) (1000000D * Math.random()));
        ans = ans.resampleWithWeights(rand);
        return ans;
    }

    public double[] getAcc() {
        return accuracy;
    }

    public void execute()
            throws Exception {
        double label[][] = new double[maxIt][InDomainU.numInstances()];
        Instances newInU = new Instances(InDomainU);
        Instances newInL = new Instances(InDomainL);
        Instances newOut = new Instances(OutDomain);
        Instances selInL = new Instances(InDomainL);
        Instances selOut = new Instances(OutDomain);
        double maxAcc = 4.9406564584124654E-324D;
        for (int i = 0; i < maxIt; i++) {
            Instances All = new Instances(selInL);
            for (int j = 0; j < selOut.numInstances(); j++) {
                All.add((Instance) selOut.instance(j).copy());
            }

            KDA kda = new KDA(All);
            kda.excute();
            newInL = kda.decInstances(selInL);
            newInU = kda.decInstances(InDomainU);
            newOut = kda.decInstances(selOut);
            Instances classInL = kda.decInstances(InDomainL);
            Instances classOut = kda.decInstances(OutDomain);
            Instances trainL = new Instances(newInL);
            for (int j = 0; j < newOut.numInstances(); j++) {
                trainL.add((Instance) newOut.instance(j).copy());
            }

            myClassifier MC = new myClassifier(trainL, Classifier.makeCopy(specific_classifier));
            double tc = MC.TestAccuracy(newInU);
            maxAcc = Math.max(tc, maxAcc);
            if (i == maxIt - 1) {
                if (intCV == maxIt) {
                    fw.write((new StringBuilder(String.valueOf(maxAcc))).append("\n").toString());
                } else {
                    fw.write((new StringBuilder(String.valueOf(tc))).append("\n").toString());
                }
            }
            double tmpInL[] = new double[newInL.numInstances()];
            double tmpOut[] = new double[newOut.numInstances()];
            for (int j = 0; j < newInU.numInstances(); j++) {
                label[i][j] = MC.classifyInstance(newInU.instance(j));
            }

            for (int j = 0; j < newInL.numInstances(); j++) {
                tmpInL[j] = MC.classifyInstance(classInL.instance(j));
            }

            for (int j = 0; j < newOut.numInstances(); j++) {
                tmpOut[j] = MC.classifyInstance(classOut.instance(j));
            }

            selOut = selectO(tmpOut, i);
            selInL = selectL(tmpInL, i);
            double acc = 0.0D;
            for (int j = 0; j < InDomainU.numInstances(); j++) {
                double lf = 1.0D;
                double rt = 1.0D;
                for (int k = i / 2; k <= i; k++) {
                    lf *= Math.pow(betaVal[k], -label[k][j]);
                    rt *= Math.pow(betaVal[k], -0.5D);
                }

                if (lf >= rt && InDomainU.instance(j).classValue() == 1.0D) {
                    acc++;
                } else if (lf < rt && InDomainU.instance(j).classValue() == 0.0D) {
                    acc++;
                }
            }

            accuracy[i] = (acc / (double) InDomainU.numInstances()) * 100D;
        }

    }

    public static void runReuters(int maxCV, double ratio)
            throws Exception {
        String testDataName = "";
        String trainDataName = "";
        String dataName = "";
        for (int i = 0; i < 3; i++) {
            if (i == 0) {
                trainDataName = "orgs vs people Train";
                testDataName = "orgs vs people test";
                dataName = "orgs vs people";
            }
            if (i == 1) {
                trainDataName = "orgs vs places train";
                testDataName = "orgs vs places test";
                dataName = "orgs vs places";
            }
            if (i == 2) {
                trainDataName = "people vs places train";
                testDataName = "people vs places test";
                dataName = "people vs places";
            }
            Instances trainData = new Instances(new BufferedReader(new FileReader((new StringBuilder("DataSet/Reuters-21578/")).append(trainDataName).append(".arff").toString())));
            trainData.setClassIndex(trainData.numAttributes() - 1);
            Instances testData = new Instances(new BufferedReader(new FileReader((new StringBuilder("DataSet/Reuters-21578/")).append(testDataName).append(".arff").toString())));
            testData.setClassIndex(testData.numAttributes() - 1);
            trainDataName = trainData.relationName();
            testDataName = testData.relationName();
            fw.write((new StringBuilder(String.valueOf(dataName))).append(" ").toString());
            KMapWeight KW = new KMapWeight(testData, trainData, maxCV, ratio);
            KW.execute();
        }

    }

    public static void runSyskillWebert(int maxCV, double ratio)
            throws Exception {
        String testDataName = "";
        String trainDataName = "";
        Scanner name = new Scanner(new File("zwj/util/UCI/name.txt"));
        int numDataSet = 0;
        int i = 0;
        int j = 0;
        while (name.hasNext()) {
            name.next();
            numDataSet++;
        }
        String dataSetName[] = new String[numDataSet];
        for (name = new Scanner(new File("zwj/util/UCI/name.txt")); name.hasNext(); ) {
            dataSetName[i++] = name.next();
        }

        for (i = 0; i < 1; i++) {
            for (j = 0; j < numDataSet; j++) {
                if (i != j) {
                    Instances trainData = new Instances(new BufferedReader(new FileReader((new StringBuilder("zwj/util/UCI/")).append(dataSetName[i]).append(".arff").toString())));
                    trainData.setClassIndex(0);
                    Instances testData = new Instances(new BufferedReader(new FileReader((new StringBuilder("zwj/util/UCI/")).append(dataSetName[j]).append(".arff").toString())));
                    testData.setClassIndex(0);
                    trainDataName = trainData.relationName();
                    testDataName = testData.relationName();
                    fw.write((new StringBuilder(String.valueOf(trainDataName))).append("vs.").append(testDataName).append(" ").toString());
                    KMapWeight KW = new KMapWeight(testData, trainData, maxCV, ratio);
                    KW.execute();
                }
            }
        }

    }

    public static void run20News(int maxCV, double ratio)
            throws Exception {
        for (int i = 0; i < 6; i++) {
            String name = "";
            if (i == 0) {
                name = "ComVsRec";
            } else if (i == 1) {
                name = "ComVsSci";
            } else if (i == 2) {
                name = "ComVstalk";
            }
            System.out.println(name);
            Instances trainData = new Instances(new BufferedReader(new FileReader((new StringBuilder("DataSet/20newsGroup/")).append(name).append("/outDomain.arff").toString())));
            trainData.setClassIndex(trainData.numAttributes() - 1);
            Instances testData = new Instances(new BufferedReader(new FileReader((new StringBuilder("DataSet/20newsGroup/")).append(name).append("/inDomain.arff").toString())));
            testData.setClassIndex(testData.numAttributes() - 1);
            fw.write((new StringBuilder(String.valueOf(name))).append(" ").toString());
            KMapWeight KW = new KMapWeight(testData, trainData, maxCV, ratio);
            KW.execute();
        }

    }

    public static void main(String args[])
            throws Exception {
        String classifierName = "";
        int maxCV = (new Integer(args[0])).intValue();
        double ratio = (new Double(args[1])).doubleValue();
        String dataName = args[2];
        System.out.println("Running....");
        fw = new FileWriter("res_KMapWeight.txt", true);
        fw.write((new StringBuilder("Number of Iterations: ")).append(maxCV).append("\n").toString());
        for (int idx = 1; idx <= 3; idx++) {
            switch (idx) {
                case 1: // '\001'
                    specific_classifier = new NaiveBayes();
                    classifierName = "NaiveBayes";
                    break;

                case 2: // '\002'
                    specific_classifier = new SMO();
                    classifierName = "SMO";
                    break;

                case 3: // '\003'
                    specific_classifier = new IBk(3);
                    classifierName = "IBk";
                    break;
            }
            fw.write((new StringBuilder(String.valueOf(classifierName))).append("\n").toString());
            if (dataName.equals("SyskillWebert")) {
                runSyskillWebert(maxCV, ratio);
            }
            if (dataName.equals("Reuters")) {
                runReuters(maxCV, ratio);
            }
            if (dataName.equals("20News")) {
                run20News(maxCV, ratio);
            }
        }

        fw.close();
    }

    private Instances[] getLU(Instances data, int numL) {
        for (int i = 0; i < data.numAttributes(); i++) {
            data.deleteWithMissing(i);
        }

        int numClasses[] = NumForEachClass(numL, data.numClasses());
        Instances L = new Instances(data, 0);
        Instances testingSet = new Instances(data, 0);
        Random rand = new Random((long) (1000000D * Math.random()));
        data.randomize(rand);
        for (int i = 0; i < data.numInstances(); i++) {
            int theClass = (int) data.instance(i).classValue();
            if (numClasses[theClass] > 0) {
                L.add(data.instance(i));
                numClasses[theClass]--;
            } else {
                testingSet.add(data.instance(i));
            }
        }

        Instances result[] = {
                L, testingSet
        };
        return result;
    }

    private int[] NumForEachClass(int total, int classes) {
        int array[] = new int[classes];
        int dim = classes;
        for (int i = 0; i < classes; i++) {
            int k = Math.round(total / dim);
            array[i] = k;
            total -= k;
            dim--;
        }

        return array;
    }

    Instances InDomainU;
    Instances InDomainL;
    double weightL[];
    Instances OutDomain;
    double weightO[];
    Instances All;
    static Classifier specific_classifier;
    int maxIt;
    double accuracy[];
    double betaVal[];
    public static FileWriter fw;
    public static int intCV = 10;

}
