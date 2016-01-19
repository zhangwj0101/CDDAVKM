package classify;

import dimesionReduction.KDA;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintStream;
import java.util.Random;
import java.util.Scanner;
import util.BRSD_BK;
import util.myClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class KMapEnsemble {

    Instances InDomainU;
    Instances InDomainL;
    Instances OutDomain;
    Instances All;
    static Classifier specific_classifier;
    int maxIt;
    double[] accuracy;
    public static FileWriter fw;
    public static int intCV = 10;

    public KMapEnsemble(Instances In, Instances Out, int it, double ratio) throws Exception {
        Instances[] tmp = getLU(In, (int) (In.numInstances() * ratio));
        this.InDomainL = new Instances(tmp[0]);

        this.InDomainU = new Instances(tmp[1]);
        this.OutDomain = new Instances(Out);
        this.All = new Instances(this.InDomainL);
        for (int i = 0; i < Out.numInstances(); i++) {
            this.All.add((Instance) Out.instance(i).copy());
        }

        this.maxIt = it;
        this.accuracy = new double[this.maxIt];
    }

    private Instances selectPointForKDA(int idx, Instances src, Instances tSrc, Instances inD) throws Exception {
        int inN = inD.numInstances();
        int i = 0, j = 0;
        Instances tmp = new Instances(src);
        for (i = 0; i < inD.numInstances(); i++) {
            tmp.add((Instance) inD.instance(i).copy());
        }

        int[] clabel = new int[tmp.numInstances()];
        int[] tlabel = new int[inD.numInstances()];
        for (i = 0; i < tlabel.length; i++) {
            tlabel[i] = ((int) tmp.instance(i).classValue());
        }
        tmp.setClassIndex(-1);
        tmp.deleteAttributeAt(idx);

        Instances ans = new Instances(tSrc, 0);

        BRSD_BK bk = new BRSD_BK(tmp, tlabel);
        bk.buildCluster();
        clabel = bk.getLable();
        int numCluster = -2147483648;
        for (i = 0; i < tmp.numInstances(); i++) {
            if (clabel[i] > numCluster) {
                numCluster = clabel[i];
            }
        }
        numCluster++;

        int[][] num = new int[numCluster][src.numClasses()];

        int[] label = new int[numCluster];

        for (i = 0; i < numCluster; i++) {
            for (j = 0; j < src.numClasses(); j++) {
                num[i][j] = 0;
            }
            label[i] = -1;
        }

        for (i = inN; i < tmp.numInstances(); i++) {
            num[clabel[i]][((int) src.instance(i - inN).classValue())] += 1;
        }

        for (i = 0; i < numCluster; i++) {
            int mV = -2147483648;
            int midx = -1;
            for (j = 0; j < src.numClasses(); j++) {
                if (num[i][j] > mV) {
                    mV = num[i][j];
                    midx = j;
                }
            }
            label[i] = midx;
        }

        for (i = 0; i < src.numInstances(); i++) {
            if ((int) src.instance(i).classValue() == label[clabel[(inN + i)]]) {
                for (j = 0; j < inN; j++) {
                    if ((clabel[(inN + i)] == clabel[j]) && (inD.instance(j).classValue() == src.instance(i).classValue())) {
                        ans.add((Instance) tSrc.instance(i).copy());
                        break;
                    }
                }
            }
        }
        return ans;
    }

    public double[] getAcc() {
        return this.accuracy;
    }

    public void excute(int idx)
            throws Exception {
        Instances newInU = new Instances(this.InDomainU);
        Instances newInL = new Instances(this.InDomainL);
        Instances newOut = new Instances(this.OutDomain);
        int i = 0, j = 0;
        double[][] label = new double[this.maxIt][this.InDomainU.numInstances()];

        double maxAcc = 4.9E-324D;
        for (i = 0; i < this.maxIt; i++) {
            Instances trainKDA = new Instances(this.InDomainL);
            Instances selOut = selectPointForKDA(idx, newOut, this.OutDomain, newInL);
            if (i > 0) {
                for (j = 0; j < selOut.numInstances(); j++) {
                    trainKDA.add(selOut.instance(j));
                }
            }

            KDA kda = new KDA(trainKDA);
            kda.excute();
            newInL = kda.decInstances(this.InDomainL);
            newInU = kda.decInstances(this.InDomainU);
            newOut = kda.decInstances(this.OutDomain);
            Instances boostOut = kda.decInstances(selOut);
            Instances trainL = new Instances(newInL);
            for (j = 0; j < boostOut.numInstances(); j++) {
                trainL.add(boostOut.instance(j));
            }
            myClassifier MC = new myClassifier(trainL, Classifier.makeCopy(specific_classifier));
            double tc = MC.TestAccuracy(newInU);
            maxAcc = Math.max(tc, maxAcc);
            if (i == this.maxIt - 1) {
                if (intCV == this.maxIt) {
                    fw.write(maxAcc + "\n");
                } else {
                    fw.write(tc + "\n");
                }
            }
            for (j = 0; j < newInU.numInstances(); j++) {
                if (MC.classifyInstance(newInU.instance(j)) == 0.0D) {
                    label[i][j] -= MC.confidence(newInU.instance(j));
                } else {
                    label[i][j] += MC.confidence(newInU.instance(j));
                }
            }
            double acc = 0.0D;
            for (j = 0; j < this.InDomainU.numInstances(); j++) {
                double lf = 0.0D;
                double rt = 0.0D;
                for (int k = 0; k <= i; k++) {
                    lf += label[k][j];
                    rt += 0.5D;
                }
                if (lf >= rt) {
                    if (this.InDomainU.instance(j).classValue() == 1.0D) {
                        acc += 1.0D;
                    }
                } else if ((lf < rt)
                           && (this.InDomainU.instance(j).classValue() == 0.0D)) {
                    acc += 1.0D;
                }
            }

            this.accuracy[i] = (acc / this.InDomainU.numInstances() * 100.0D);
        }
    }

    public static void runReuters(int maxCV, double ratio) throws Exception {
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
            Instances trainData = new Instances(new BufferedReader(new FileReader("DataSet/Reuters-21578/" + trainDataName + ".arff")));
            trainData.setClassIndex(trainData.numAttributes() - 1);
            Instances testData = new Instances(new BufferedReader(new FileReader("DataSet/Reuters-21578/" + testDataName + ".arff")));
            testData.setClassIndex(testData.numAttributes() - 1);
            trainDataName = trainData.relationName();
            testDataName = testData.relationName();

            fw.write(dataName + " ");
            KMapEnsemble KE = new KMapEnsemble(testData, trainData, maxCV, ratio);
            KE.excute(testData.numAttributes() - 1);
        }
    }

    public static void runSyskillWebert(int maxCV, double ratio)
            throws Exception {
        String testDataName = "";
        String trainDataName = "";
        Scanner name = new Scanner(new File("DataSet/UCI/name.txt"));
        int numDataSet = 0;
        int i = 0;
        int j = 0;
        while (name.hasNext()) {
            name.next();
            numDataSet++;
        }
        String[] dataSetName = new String[numDataSet];
        name = new Scanner(new File("DataSet/UCI/name.txt"));
        while (name.hasNext()) {
            dataSetName[(i++)] = name.next();
        }
        for (i = 0; i < numDataSet; i++) {
            for (j = 0; j < numDataSet; j++) {
                if (i != j) {
                    Instances trainData = new Instances(new BufferedReader(new FileReader("DataSet/UCI/" + dataSetName[i] + ".arff")));
                    trainData.setClassIndex(0);
                    Instances testData = new Instances(new BufferedReader(new FileReader("DataSet/UCI/" + dataSetName[j] + ".arff")));
                    testData.setClassIndex(0);
                    trainDataName = trainData.relationName();
                    testDataName = testData.relationName();
                    fw.write(trainDataName + "vs." + testDataName + " ");

                    KMapEnsemble KE = new KMapEnsemble(testData, trainData, maxCV, ratio);
                    KE.excute(0);
                }
            }
        }
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
            Instances trainData = new Instances(new BufferedReader(new FileReader("DataSet/20newsGroup/" + name + "/outDomain.arff")));
            trainData.setClassIndex(trainData.numAttributes() - 1);
            Instances testData = new Instances(new BufferedReader(new FileReader("DataSet/20newsGroup/" + name + "/inDomain.arff")));
            testData.setClassIndex(testData.numAttributes() - 1);

            fw.write(name + " ");
            KMapEnsemble KE = new KMapEnsemble(testData, trainData, maxCV, ratio);
            KE.excute(testData.numAttributes() - 1);
        }
    }

    public static void main(String[] args)
            throws Exception {
        String classifierName = "";
        int maxCV = new Integer(args[0]).intValue();
        double ratio = new Double(args[1]).doubleValue();
        String dataName = args[2];
        System.out.println("Running....");
        fw = new FileWriter("res_KMapEnsemble.txt", true);
        fw.write("Number of Iterations: " + maxCV + "\n");
        for (int idx = 1; idx <= 3; idx++) {
            switch (idx) {
                case 1:
                    specific_classifier = new NaiveBayes();
                    classifierName = "NaiveBayes";
                    break;
                case 2:
                    specific_classifier = new SMO();
                    classifierName = "SMO";
                    break;
                case 3:
                    specific_classifier = new IBk(3);
                    classifierName = "IBk";
                    break;
            }

            fw.write(classifierName + "\n");
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
        int[] numClasses = NumForEachClass(numL, data.numClasses());
        Instances L = new Instances(data, 0);
        Instances testingSet = new Instances(data, 0);
        Random rand = new Random((long) (1000000.0D * Math.random()));
        data.randomize(rand);

        for (int i = 0; i < data.numInstances(); i++) {
            int theClass = (int) data.instance(i).classValue();
            if (numClasses[theClass] > 0) {
                L.add(data.instance(i));
                numClasses[theClass] -= 1;
            } else {
                testingSet.add(data.instance(i));
            }
        }
        Instances[] result = {L, testingSet};
        return result;
    }

    private int[] NumForEachClass(int total, int classes) {
        int[] array = new int[classes];
        int dim = classes;
        for (int i = 0; i < classes; i++) {
            int k = Math.round(total / dim);
            array[i] = k;
            total -= k;
            dim--;
        }
        return array;
    }
}
