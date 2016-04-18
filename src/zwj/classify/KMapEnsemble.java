// Decompiled by Jad v1.5.8g. Copyright 2001 Pavel Kouznetsov.
// Jad home page: http://www.kpdus.com/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   KMapEnsemble.java
package zwj.classify;

import zwj.dimesionReduction.KDA;

import java.io.*;
import java.util.*;

import zwj.util.BRSD_BK;
import zwj.util.myClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class KMapEnsemble {

    private static final String BASE = "C:/CDDAVKM/";

    public KMapEnsemble(Instances In, Instances Out, int it, double ratio)
            throws Exception {
        Instances tmp[] = getLU(In, (int) ((double) In.numInstances() * ratio));
        InDomainL = new Instances(tmp[0]); //训练
        InDomainU = new Instances(tmp[1]); //测试
        OutDomain = new Instances(Out); //源领域训练
        All = new Instances(InDomainL);
        for (int i = 0; i < Out.numInstances(); i++) {
            All.add((Instance) Out.instance(i).copy());
        }

        maxIt = it;
        accuracy = new double[maxIt];
    }


    private Instances selectPointForKDA(int idx, Instances src, Instances tSrc, Instances inD)
            throws Exception {
        int inN = inD.numInstances();
        Instances tmp = new Instances(src);
        for (int i = 0; i < inD.numInstances(); i++) {
            tmp.add((Instance) inD.instance(i).copy());
        }

        int clabel[] = new int[tmp.numInstances()];
        int tlabel[] = new int[inD.numInstances()];
        for (int i = 0; i < tlabel.length; i++) {
            tlabel[i] = (int) tmp.instance(i).classValue();
        }

        tmp.setClassIndex(-1);
        tmp.deleteAttributeAt(idx);
        Instances ans = new Instances(tSrc, 0);
        BRSD_BK bk = new BRSD_BK(tmp, tlabel);
        bk.buildCluster();
        clabel = bk.getLable();
        int numCluster = 0x80000000;
        for (int i = 0; i < tmp.numInstances(); i++) {
            if (clabel[i] > numCluster) {
                numCluster = clabel[i];
            }
        }

        int num[][] = new int[++numCluster][src.numClasses()];
        int label[] = new int[numCluster];
        for (int i = 0; i < numCluster; i++) {
            for (int j = 0; j < src.numClasses(); j++) {
                num[i][j] = 0;
            }

            label[i] = -1;
        }

        for (int i = inN; i < tmp.numInstances(); i++) {
            num[clabel[i]][(int) src.instance(i - inN).classValue()]++;
        }

        for (int i = 0; i < numCluster; i++) {
            int mV = 0x80000000;
            int midx = -1;
            for (int j = 0; j < src.numClasses(); j++) {
                if (num[i][j] > mV) {
                    mV = num[i][j];
                    midx = j;
                }
            }

            label[i] = midx;
        }

        for (int i = 0; i < src.numInstances(); i++) {
            if ((int) src.instance(i).classValue() == label[clabel[inN + i]]) {
                for (int j = 0; j < inN; j++) {
                    if (clabel[inN + i] != clabel[j] || inD.instance(j).classValue() != src.instance(i).classValue()) {
                        continue;
                    }
                    ans.add((Instance) tSrc.instance(i).copy());
                    break;
                }

            }
        }

        return ans;
    }

    public double[] getAcc() {
        return accuracy;
    }

    public void excute(int idx)
            throws Exception {
        Instances newInU = new Instances(InDomainU);
        Instances newInL = new Instances(InDomainL);
        double averageAcc = 0.0;
        Instances newOut = new Instances(OutDomain);
        //无监督训练数据
        Instances unsperTrain = new Instances(newInL);
        Node[] testRightCount = new Node[newInU.numInstances()];
        for (int i = 0; i < newInU.numInstances(); i++) {
            testRightCount[i] = new Node(i, 0);
        }
        double label[][] = new double[maxIt][InDomainU.numInstances()];
        double maxAcc = 4.9406564584124654E-324D;
        for (int i = 0; i < maxIt; i++) {
//            System.out.println(i + " start");
            Instances trainKDA = new Instances(InDomainL);
            Instances selOut = selectPointForKDA(idx, newOut, OutDomain, newInL);
            if (i > 0) {
                for (int j = 0; j < selOut.numInstances(); j++) {
                    trainKDA.add(selOut.instance(j));
                }
            }
            KDA kda = new KDA(trainKDA);
            kda.excute();
            newInL = kda.decInstances(InDomainL);
            newInU = kda.decInstances(InDomainU);
            newOut = kda.decInstances(OutDomain);
            Instances boostOut = kda.decInstances(selOut);
            Instances trainL = new Instances(newInL);
            for (int j = 0; j < boostOut.numInstances(); j++) {
                trainL.add(boostOut.instance(j));
            }

            myClassifier MC = new myClassifier(trainL, Classifier.makeCopy(specific_classifier));
            double tc = MC.TestAccuracy(newInU);
            averageAcc += tc;
            maxAcc = Math.max(tc, maxAcc);
            if (i == maxIt - 1) {
                if (intCV == maxIt) {
                    fw.write((new StringBuilder(String.valueOf(maxAcc))).append(" MAx \n").toString());
                } else {
                    fw.write((new StringBuilder(String.valueOf(tc))).append(" TC \n").toString());
                }
            }
            for (int j = 0; j < newInU.numInstances(); j++) {
                if (MC.classifyInstance(newInU.instance(j)) == 0.0D) {
                    label[i][j] -= MC.confidence(newInU.instance(j));
                } else {
                    label[i][j] += MC.confidence(newInU.instance(j));
                }
            }

            double acc = 0.0D;
            for (int j = 0; j < InDomainU.numInstances(); j++) {
                double lf = 0.0D;
                double rt = 0.0D;
                for (int k = 0; k <= i; k++) {
                    lf += label[k][j];
                    rt += 0.5D;
                }

                if (lf >= rt) {
                    if (InDomainU.instance(j).classValue() == 1.0D) {
                        testRightCount[j].count++;
                        acc++;
                    }
                } else if (lf < rt && InDomainU.instance(j).classValue() == 0.0D) {
                    testRightCount[j].count++;
                    acc++;
                }
            }
            System.out.printf("iter:%d\t%f\t%f\n", i, tc, maxAcc);
            accuracy[i] = (acc / (double) InDomainU.numInstances()) * 100D;
        }
        System.out.printf("average: %f\n", averageAcc / maxIt);

        Arrays.sort(testRightCount, new Comparator<Node>() {
            @Override
            public int compare(Node o1, Node o2) {
                return o2.count - o1.count;
            }
        });

        for (int i = 1; i <= 4; i++) {
            Instances originInL = new Instances(InDomainL);
            for (int j = 0; j < InDomainU.numInstances() * i * 0.1; j++) {
                originInL.add(InDomainU.instance(j));
            }
            myClassifier MC = new myClassifier(originInL, Classifier.makeCopy(specific_classifier));
            double tc = MC.TestAccuracy(InDomainU);
            System.out.printf("it %d\tunsu : %.8f\n", i, tc);
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
            Instances trainData = new Instances(new BufferedReader(new FileReader((new StringBuilder(BASE + "DataSet/Reuters-21578/")).append(trainDataName).append(".arff").toString())));
            trainData.setClassIndex(trainData.numAttributes() - 1);
            Instances testData = new Instances(new BufferedReader(new FileReader((new StringBuilder(BASE + "DataSet/Reuters-21578/")).append(testDataName).append(".arff").toString())));
            testData.setClassIndex(testData.numAttributes() - 1);
            trainDataName = trainData.relationName();
            testDataName = testData.relationName();
            fw.write((new StringBuilder(String.valueOf(dataName))).append(" ").toString());
            KMapEnsemble KE = new KMapEnsemble(testData, trainData, maxCV, ratio);
            KE.excute(testData.numAttributes() - 1);
        }

    }

    public static void runSyskillWebert(int maxCV, double ratio)
            throws Exception {
        String testDataName = "";
        String trainDataName = "";
        Scanner name = new Scanner(new File(BASE + "DataSet/UCI/name.txt"));
        int numDataSet = 0;
        int i = 0;
        int j = 0;
        while (name.hasNext()) {
            name.next();
            numDataSet++;
        }
        String dataSetName[] = new String[numDataSet];
        for (name = new Scanner(new File(BASE + "DataSet/UCI/name.txt")); name.hasNext(); ) {
            dataSetName[i++] = name.next();
        }

        for (i = 0; i < numDataSet; i++) {
            for (j = 0; j < numDataSet; j++) {
                if (i != j) {
                    Instances trainData = new Instances(new BufferedReader(new FileReader((new StringBuilder(BASE + "DataSet/UCI/")).append(dataSetName[i]).append(".arff").toString())));
                    trainData.setClassIndex(0);
                    Instances testData = new Instances(new BufferedReader(new FileReader((new StringBuilder(BASE + "DataSet/UCI/")).append(dataSetName[j]).append(".arff").toString())));
                    testData.setClassIndex(0);
                    trainDataName = trainData.relationName();
                    testDataName = testData.relationName();
                    String res = (new StringBuilder(String.valueOf(trainDataName))).append(" vs ").append(testDataName).append(" : ").toString();
                    System.out.println(res);
                    fw.write(res);
                    KMapEnsemble KE = new KMapEnsemble(testData, trainData, maxCV, ratio);
                    KE.excute(0);
                }
            }
        }

    }

    public static void run20News_ZWJ(int maxCV, double ratio)
            throws Exception {
        String name = "AD";
        fw.write(name);

        Instances trainData = new Instances(new BufferedReader(new FileReader("C:\\20NG\\ct_outDomain.arff")));
        trainData.setClassIndex(trainData.numAttributes() - 1);
        Instances testData = new Instances(new BufferedReader(new FileReader("C:\\20NG\\ct_inDomain.arff")));
        testData.setClassIndex(testData.numAttributes() - 1);
        fw.write((new StringBuilder(String.valueOf(name))).append(" ").toString());
        KMapEnsemble KE = new KMapEnsemble(testData, trainData, maxCV, ratio);
        KE.excute(testData.numAttributes() - 1);


    }

    public static void runReuters_ZWJ(int maxCV, double ratio, String srcPath, String tarPath)
            throws Exception {
        System.out.println("Running....");
        fw = new FileWriter("runReuters_ZWJ.txt", true);
        specific_classifier = new NaiveBayes();
        Instances trainData = new Instances(new BufferedReader(new FileReader(srcPath)));
        trainData.setClassIndex(trainData.numAttributes() - 1);
        Instances testData = new Instances(new BufferedReader(new FileReader(tarPath)));
        testData.setClassIndex(testData.numAttributes() - 1);
        String trainDataName = trainData.relationName();
        String testDataName = testData.relationName();
        String res = (new StringBuilder(String.valueOf(trainDataName))).append(" vs ").append(testDataName).append(" : ").toString();
        System.out.println(res);
        fw.write(res);
        KMapEnsemble KE = new KMapEnsemble(testData, trainData, maxCV, ratio);
        KE.excute(testData.numAttributes() - 1);
    }

    public static void run20News(int maxCV, double ratio)
            throws Exception {
        for (int i = 0; i < 3; i++) {
            String name = "";
            if (i == 0) {
                name = "ComVsRec";
            } else if (i == 1) {
                name = "ComVsSci";
            } else if (i == 2) {
                name = "ComVstalk";
            }
            fw.write(name);
            Instances trainData = new Instances(new BufferedReader(new FileReader((new StringBuilder(BASE + "DataSet/20newsGroup/")).append(name).append("/outDomain.arff").toString())));
            trainData.setClassIndex(trainData.numAttributes() - 1);
            Instances testData = new Instances(new BufferedReader(new FileReader((new StringBuilder(BASE + "DataSet/20newsGroup/")).append(name).append("/inDomain.arff").toString())));
            testData.setClassIndex(testData.numAttributes() - 1);
            fw.write((new StringBuilder(String.valueOf(name))).append(" ").toString());
            KMapEnsemble KE = new KMapEnsemble(testData, trainData, maxCV, ratio);
            KE.excute(testData.numAttributes() - 1);
        }

    }


    public static void origin() throws Exception {
        String classifierName = "";
//        int maxCV = (new Integer(args[0])).intValue();
//        double ratio = (new Double(args[1])).doubleValue();
//        String dataName = args[2];
        int maxCV = 10;
        double ratio = 0.1;
        String dataName = "20News";
        System.out.println("Running....");
        fw = new FileWriter("res_KMapEnsemble.txt", true);
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
            fw.write("\n");
        }

        fw.close();
    }

    public static void main(String args[]) throws Exception {
        String classifierName = "";
        int maxCV = 10;
        double ratio = 0.1;
        String dataName = "SyskillWebert";
        System.out.println("Running....");
        fw = new FileWriter("res_KMapEnsemble.txt", true);
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
            fw.write("\n");
        }

        fw.close();
    }

    /**
     * 挑选训练集和测试集
     *
     * @param data
     * @param numL
     * @return
     */
    private Instances[] getLU(Instances data, int numL) {
        /**
         * 保证每一列都存在
         */
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
    Instances OutDomain;
    Instances All;
    static Classifier specific_classifier;
    int maxIt;
    double accuracy[];
    public static FileWriter fw;
    public static int intCV = 10;

}

class Node {
    int idx;
    int count = 0;

    public Node(int idx, int count) {
        this.idx = idx;
        this.count = count;
    }

    @Override
    public String toString() {
        return "Node{" +
                "idx=" + idx +
                ", count=" + count +
                '}';
    }
}
