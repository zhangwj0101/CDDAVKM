// Decompiled by Jad v1.5.8g. Copyright 2001 Pavel Kouznetsov.
// Jad home page: http://www.kpdus.com/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   KDA.java
package dimesionReduction;

import java.io.*;
import java.util.Arrays;

import util.MyMath;
import weka.core.*;
import weka.core.matrix.EigenvalueDecomposition;
import weka.core.matrix.Matrix;

public class KDA {

    public KDA(Instances data) {
        data.sort(data.numAttributes() - 1);
        srcData = new Instances(data);
        num = new int[data.numClasses()];
        kernelM = new Matrix(data.numInstances(), data.numInstances());
        W = new Matrix(data.numInstances(), data.numInstances());
    }

    public void excute() {
        buildKM();
        calSpNum();
        buildW();
        calEig();
        comNewKennel();
        calBeta();
        calAphal();
    }

    private void buildKM() {
        sumK = new Matrix(srcData.numInstances(), srcData.numInstances());
        Matrix ones = new Matrix(srcData.numInstances(), srcData.numInstances());
        omit = 4.9406564584124654E-324D;
        MyMath mh = new MyMath();
        for (int i = 0; i < srcData.numInstances(); i++) {
            for (int j = 0; j < srcData.numInstances(); j++) {
                double v = MyMath.CalSim(srcData.instance(i), srcData.instance(j));
                if (srcData.instance(i).classValue() != srcData.instance(j).classValue()) {
                    omit = Math.max(v, omit);
                }
            }

        }

        for (int i = 0; i < srcData.numInstances(); i++) {
            for (int j = 0; j < srcData.numInstances(); j++) {
                double val = Math.exp(-MyMath.CalSim(srcData.instance(i), srcData.instance(j)) / omit);
                kernelM.set(i, j, val);
                ones.set(i, j, 1.0D / (double) srcData.numInstances());
            }

        }

        sumK = ones.times(kernelM);
        kernelM = kernelM.minus(sumK.transpose()).minus(sumK).plus(sumK.times(ones));
    }

    private void calSpNum() {

        Arrays.fill(num, 0);

        for (int i = 0; i < srcData.numInstances(); i++) {
            num[(int) srcData.instance(i).classValue()]++;
        }

    }

    private void buildW() {
        int pt = 0;
        for (int i = 0; i < srcData.numInstances(); i++) {
            for (int j = 0; j < srcData.numInstances(); j++) {
                W.set(i, j, 0.0D);
            }

        }

        for (int i = 0; i < srcData.numClasses(); i++) {
            for (int j = 0; j < num[i]; j++) {
                for (int k = 0; k < num[i]; k++) {
                    W.set(pt + j, pt + k, 1.0D / (double) num[i]);
                }

            }

            pt += num[i];
        }

    }

    private void calEig() {
        EigenvalueDecomposition ED = kernelM.eig();
        eigVal = ED.getD();
        eigVec = ED.getV();

        MyMath.quickSort(eigVal, eigVec);
        rankKM = 0;
        double minV = eigVal.get(0, 0) / 1000D;
        for (int i = 0; i < eigVal.getRowDimension(); i++) {
            if (eigVal.get(i, i) < minV) {
                break;
            }
            rankKM++;
        }

        rankKM = Math.min(rankKM, srcData.numInstances());
    }

    private void comNewKennel() {
        eigVal = eigVal.getMatrix(0, rankKM - 1, 0, rankKM - 1);
        eigVec = eigVec.getMatrix(0, eigVec.getRowDimension() - 1, 0, rankKM - 1);
        kernelM = eigVec.times(eigVal).times(eigVec.transpose());
    }


    private void calBeta() {
        Matrix newT = eigVec.transpose().times(W).times(eigVec);
        EigenvalueDecomposition ED = newT.eig();
        betaVal = ED.getD();
        betaVec = ED.getV();
        MyMath.quickSort(betaVal, betaVec);
    }

    private void calAphal() {
        Matrix tmp = eigVec.times(eigVal.inverse()).times(betaVec);
        aphal = new Matrix(srcData.numInstances(), rankKM);
        for (int i = 0; i < rankKM; i++) {
            Matrix tv = tmp.getMatrix(0, srcData.numInstances() - 1, i, i);
            double val = Math.sqrt(tv.transpose().times(kernelM).times(tv).get(0, 0));
            for (int j = 0; j < srcData.numInstances(); j++) {
                aphal.set(j, i, tmp.get(j, i) / val);
            }

        }
    }

    public Instances decInstances(Instances newData) {
        MyMath mh = new MyMath();
        Matrix newKernel = new Matrix(newData.numInstances(), srcData.numInstances());
        for (int i = 0; i < newData.numInstances(); i++) {
            double sum = 0.0D;
            for (int j = 0; j < srcData.numInstances(); j++) {
                double val = Math.exp(-MyMath.CalSim(newData.instance(i), srcData.instance(j)) / omit);
                newKernel.set(i, j, val);
                sum += val;
            }

            for (int j = 0; j < srcData.numInstances(); j++) {
                newKernel.set(i, j, newKernel.get(i, j) - sum / (double) srcData.numInstances());
            }

        }

        Matrix newSum = sumK.getMatrix(0, 0, 0, sumK.getRowDimension() - 1);
        double sumV = 0.0D;
        for (int i = 0; i < sumK.getRowDimension(); i++) {
            sumV += newSum.get(0, i);
        }

        for (int i = 0; i < sumK.getRowDimension(); i++) {
            newSum.set(0, i, newSum.get(0, i) - sumV / (double) srcData.numInstances());
        }

        for (int i = 0; i < newData.numInstances(); i++) {
            for (int j = 0; j < srcData.numInstances(); j++) {
                newKernel.set(i, j, newKernel.get(i, j) - newSum.get(0, j));
            }

        }

        newKernel = newKernel.times(aphal);
        FastVector fv = new FastVector();
        for (int i = 0; i < newKernel.getColumnDimension(); i++) {
            fv.addElement(new Attribute((new StringBuilder("F")).append(i).toString()));
        }

        FastVector classAttribute = new FastVector();
        for (int i = 0; i < newData.numClasses(); i++) {
            classAttribute.addElement((new StringBuilder()).append(i).toString());
        }

        fv.addElement(new Attribute("class", classAttribute));
        Instances disData = new Instances((new StringBuilder("RD_")).append(srcData.relationName()).toString(), fv, newData.numInstances());
        for (int i = 0; i < newKernel.getRowDimension(); i++) {
            Instance f = new Instance(newKernel.getColumnDimension() + 1);
            int j;
            for (j = 0; j < newKernel.getColumnDimension(); j++) {
                f.setValue(j, newKernel.get(i, j));
            }

            f.setValue(j, (int) newData.instance(i).classValue());
            disData.add(f);
        }

        disData.setClassIndex(newKernel.getColumnDimension());
        return disData;
    }


    public static void main(String args[])
            throws Exception {
        Instances trainData = new Instances(new BufferedReader(new FileReader("F:/workspace/Python/Data/Arff/All.arff")));
        trainData.setClassIndex(trainData.numAttributes() - 1);
        Instances testData = new Instances(new BufferedReader(new FileReader("F:/workspace/Python/Data/Arff/All.arff")));
        testData.setClassIndex(testData.numAttributes() - 1);
        KDA kda = new KDA(trainData);
        kda.excute();
        Instances ans = kda.decInstances(testData);
        FileWriter Fw = new FileWriter("F:/workspace/Python/Data/KDA/All.arff");
        Fw.write(ans.toString());
        Fw.close();
    }

    private Instances srcData;
    private Matrix kernelM;
    private Matrix W;
    Matrix sumK;
    private int num[];
    private Matrix eigVec;
    private Matrix eigVal;
    private int rankKM;
    private Matrix betaVec;
    private Matrix betaVal;
    private Matrix aphal;
    private double omit;
}
