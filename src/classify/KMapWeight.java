package classify;

import dimesionReduction.KDA;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintStream;
import java.util.Random;
import java.util.Scanner;
import util.myClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class KMapWeight {

    Instances InDomainU;
    Instances InDomainL;
    double[] weightL;
    Instances OutDomain;
    double[] weightO;
    Instances All;
    static Classifier specific_classifier;
    int maxIt;
    double[] accuracy;
    double[] betaVal;
    public static FileWriter fw;
    public static int intCV = 10;

    public KMapWeight(Instances In, Instances Out, int it, double ratio) throws Exception {
        Instances[] tmp = getLU(In, (int) (In.numInstances() * ratio));
        this.InDomainU = new Instances(tmp[1]);
        this.InDomainL = new Instances(tmp[0]);

        this.OutDomain = new Instances(Out);

        this.All = new Instances(this.InDomainL);
        for (int i = 0; i < Out.numInstances(); i++) {
            this.All.add((Instance) Out.instance(i).copy());
        }

        this.maxIt = it;
        this.accuracy = new double[this.maxIt];
        this.betaVal = new double[this.maxIt];
        this.weightL = new double[this.InDomainL.numInstances()];
        this.weightO = new double[this.OutDomain.numInstances()];
        for (int i = 0; i < this.InDomainL.numInstances(); i++) {
            this.weightL[i] = 1.0D;
        }
        for (int i = 0; i < this.OutDomain.numInstances(); i++) {
            this.weightO[i] = 1.0D;
        }
    }

    private Instances selectL(double[] newIn, int curIt)
            throws Exception {
        Instances ans = new Instances(this.InDomainL);
        int i = 0, j = 0;
        double sumW = 0.0D;
        for (i = 0; i < newIn.length; i++) {
            sumW += this.weightL[i];
        }

        double err = 0.0D;
        for (i = 0; i < newIn.length; i++) {
            err += this.weightL[i] * Math.abs(newIn[i] - this.InDomainL.instance(i).classValue());
        }
        err /= sumW;

        double betaT = err / (1.0D - err);
        if (betaT == 0.0D) {
            betaT = 1.E-005D;
        }
        this.betaVal[curIt] = betaT;
        for (i = 0; i < newIn.length; i++) {
            this.weightL[i] *= Math.pow(betaT, -Math.abs(newIn[i] - this.InDomainL.instance(i).classValue()));
            ans.instance(i).setWeight(this.weightL[i]);
        }
        Random rand = new Random((long) (1000000.0D * Math.random()));
        ans = ans.resampleWithWeights(rand);
        return ans;
    }

    private Instances selectO(double[] newOut, int curIt)
            throws Exception {
        Instances ans = new Instances(this.OutDomain);
        double beta = 1.0D / (1.0D + Math.sqrt(2.0D * Math.log(newOut.length / this.maxIt)));

        for (int i = 0; i < newOut.length; i++) {
            this.weightO[i] *= Math.pow(beta, Math.abs(newOut[i] - this.OutDomain.instance(i).classValue()));
            ans.instance(i).setWeight(this.weightO[i]);
        }
        Random rand = new Random((long) (1000000.0D * Math.random()));
        ans = ans.resampleWithWeights(rand);
        return ans;
    }

    public double[] getAcc() {
        return this.accuracy;
    }

    public void execute()
            throws Exception {
        double[][] label = new double[this.maxIt][this.InDomainU.numInstances()];
        int i = 0, j = 0;
        Instances newInU = new Instances(this.InDomainU);
        Instances newInL = new Instances(this.InDomainL);
        Instances newOut = new Instances(this.OutDomain);

        Instances selInL = new Instances(this.InDomainL);
        Instances selOut = new Instances(this.OutDomain);

        double maxAcc = 4.9E-324D;
        for (i = 0; i < this.maxIt; i++) {
            Instances All = new Instances(selInL);
            for (j = 0; j < selOut.numInstances(); j++) {
                All.add((Instance) selOut.instance(j).copy());
            }
            KDA kda = new KDA(All);
            kda.excute();

            newInL = kda.decInstances(selInL);
            newInU = kda.decInstances(this.InDomainU);
            newOut = kda.decInstances(selOut);
            Instances classInL = kda.decInstances(this.InDomainL);
            Instances classOut = kda.decInstances(this.OutDomain);

            Instances trainL = new Instances(newInL);
            for (j = 0; j < newOut.numInstances(); j++) {
                trainL.add((Instance) newOut.instance(j).copy());
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

            double[] tmpInL = new double[newInL.numInstances()];
            double[] tmpOut = new double[newOut.numInstances()];
            for (j = 0; j < newInU.numInstances(); j++) {
                label[i][j] = MC.classifyInstance(newInU.instance(j));
            }
            for (j = 0; j < newInL.numInstances(); j++) {
                tmpInL[j] = MC.classifyInstance(classInL.instance(j));
            }
            for (j = 0; j < newOut.numInstances(); j++) {
                tmpOut[j] = MC.classifyInstance(classOut.instance(j));
            }

            selOut = selectO(tmpOut, i);
            selInL = selectL(tmpInL, i);

            double acc = 0.0D;
            for (j = 0; j < this.InDomainU.numInstances(); j++) {
                double lf = 1.0D;
                double rt = 1.0D;
                for (int k = i / 2; k <= i; k++) {
                    lf *= Math.pow(this.betaVal[k], -label[k][j]);
                    rt *= Math.pow(this.betaVal[k], -0.5D);
                }
                if ((lf >= rt) && (this.InDomainU.instance(j).classValue() == 1.0D)) {
                    acc += 1.0D;
                } else if ((lf < rt) && (this.InDomainU.instance(j).classValue() == 0.0D)) {
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
            KMapWeight KW = new KMapWeight(testData, trainData, maxCV, ratio);
            KW.execute();
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
        for (i = 0; i < 1; i++) {
            for (j = 0; j < numDataSet; j++) {
                if (i != j) {
                    Instances trainData = new Instances(new BufferedReader(new FileReader("DataSet/UCI/" + dataSetName[i] + ".arff")));
                    trainData.setClassIndex(0);
                    Instances testData = new Instances(new BufferedReader(new FileReader("DataSet/UCI/" + dataSetName[j] + ".arff")));
                    testData.setClassIndex(0);
                    trainDataName = trainData.relationName();
                    testDataName = testData.relationName();
                    fw.write(trainDataName + "vs." + testDataName + " ");

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
            Instances trainData = new Instances(new BufferedReader(new FileReader("DataSet/20newsGroup/" + name + "/outDomain.arff")));
            trainData.setClassIndex(trainData.numAttributes() - 1);
            Instances testData = new Instances(new BufferedReader(new FileReader("DataSet/20newsGroup/" + name + "/inDomain.arff")));
            testData.setClassIndex(testData.numAttributes() - 1);

            fw.write(name + " ");
            KMapWeight KW = new KMapWeight(testData, trainData, maxCV, ratio);
            KW.execute();
        }
    }

    public static void main(String[] args)
            throws Exception {
        String classifierName = "";
        int maxCV = new Integer(args[0]).intValue();
        double ratio = new Double(args[1]).doubleValue();
        String dataName = args[2];
        System.out.println("Running....");
        fw = new FileWriter("res_KMapWeight.txt", true);
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
