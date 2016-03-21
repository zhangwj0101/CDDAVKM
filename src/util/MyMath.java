// Decompiled by Jad v1.5.8g. Copyright 2001 Pavel Kouznetsov.
// Jad home page: http://www.kpdus.com/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   MyMath.java
package util;

import java.io.PrintStream;
import weka.core.*;
import weka.core.matrix.Matrix;

public class MyMath {

    public MyMath() {
    }

    public static int[] sort_index(double array[]) {
        int index[] = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            int max = 0;
            for (int j = 1; j < array.length; j++) {
                if (array[j] > array[max]) {
                    max = j;
                }
            }

            index[i] = max;
            array[max] = -1D;
        }

        return index;
    }

    public static int[] sort_insert(int d[]) {
        int length = d.length;
        for (int i = 1; i < length; i++) {
            for (int j = i; j > 0; j--) {
                if (d[j] < d[j - 1]) {
                    int temp = d[j];
                    d[j] = d[j - 1];
                    d[j - 1] = temp;
                }
            }

        }

        return d;
    }

    public static double[] sort_insert(double d[]) {
        int length = d.length;
        for (int i = 1; i < length; i++) {
            for (int j = i; j > 0; j--) {
                if (d[j] < d[j - 1]) {
                    double temp = d[j];
                    d[j] = d[j - 1];
                    d[j - 1] = temp;
                }
            }

        }

        return d;
    }

    public static void output(int d[]) {
        System.out.println("\n\n^^^^^^^^^^^^");
        for (int i = 0; i < d.length; i++) {
            System.out.println(d[i]);
        }

        System.out.println("^^^^^^^^^^^^\n\n");
    }

    public static void output(double d[]) {
        System.out.println("\n\n^^^^^^^^^^^^");
        for (int i = 0; i < d.length; i++) {
            System.out.println(d[i]);
        }

        System.out.println("^^^^^^^^^^^^\n\n");
    }

    public static Instances RD(int feature[], Instances data) {
        int fn = feature.length;
        FastVector fv = new FastVector();
        for (int idx = 0; idx < fn; idx++) {
            fv.addElement(new Attribute((new StringBuilder("F")).append(idx).toString()));
        }

        FastVector classAttribute = new FastVector();
        for (int idx = 0; idx < data.numClasses(); idx++) {
            classAttribute.addElement((new StringBuilder()).append(idx).toString());
        }

        fv.addElement(new Attribute("class", classAttribute));
        Instances ND = new Instances((new StringBuilder("RD_")).append(data.relationName()).toString(), fv, data.numInstances());
        for (int idx = 0; idx < data.numInstances(); idx++) {
            Instance f = new Instance(fn + 1);
            double array[] = data.instance(idx).toDoubleArray();
            int fp;
            for (fp = 0; fp < fn; fp++) {
                f.setValue(fp, array[feature[fp]]);
            }

            f.setValue(fp, data.instance(idx).classValue());
            ND.add(f);
        }

        ND.setClassIndex(fn);
        return ND;
    }

    public static boolean isIn(int key, int array[], int n) {
        for (int i = 0; i < array.length && i < n; i++) {
            if (key == array[i]) {
                return true;
            }
        }

        return false;
    }

    private static void runSort(Matrix A, Matrix B, int left, int right) {
        int i = left;
        int j = right;
        double middle = A.get(A.getColumnDimension() / 2, A.getColumnDimension() / 2);
        do {
            while (A.get(i, i) > middle && i < right) {
                i++;
            }
            for (; A.get(j, j) < middle && j > left; j--);
            if (i <= j) {
                double temp = A.get(i, i);
                A.set(i, i, A.get(j, j));
                A.set(j, j, temp);
                for (int k = 0; k < B.getRowDimension(); k++) {
                    temp = B.get(k, i);
                    B.set(k, i, B.get(k, j));
                    B.set(k, j, temp);
                }

                i++;
                j--;
            }
        } while (i <= j);
        if (left < j) {
            runSort(A, B, left, j);
        }
        if (right > i) {
            runSort(A, B, i, right);
        }
    }

    public static void quickSort(Matrix A, Matrix B) {
        runSort(A, B, 0, A.getRowDimension() - 1);
    }

    public static double cmpP(Instances In, Instances Out) {
        double A[] = new double[In.numAttributes()];
        double B[] = new double[Out.numAttributes()];
        for (int i = 0; i < In.numAttributes() - 1; i++) {
            A[i] = 0.0D;
        }

        for (int i = 0; i < Out.numAttributes() - 1; i++) {
            B[i] = 0.0D;
        }

        for (int j = 0; j < In.numAttributes() - 1; j++) {
            for (int i = 0; i < In.numInstances(); i++) {
                A[j] += In.instance(i).value(j);
            }

            A[j] /= In.numInstances();
        }

        for (int j = 0; j < Out.numAttributes() - 1; j++) {
            for (int i = 0; i < Out.numInstances(); i++) {
                B[j] += Out.instance(i).value(j);
            }

            B[j] /= Out.numInstances();
        }

        double r = 0.0D;
        for (int i = 0; i < In.numAttributes() - 1; i++) {
            r += (A[i] - B[i]) * (A[i] - B[i]);
        }

        return r;
    }

    public static double CalSim(Instance A, Instance B) {
        double Ar[] = A.toDoubleArray();
        double Br[] = B.toDoubleArray();
        double r = 0.0D;
        int fNum = A.numAttributes() - 1;
        for (int i = 0; i < fNum; i++) {
            r += (Ar[i] - Br[i]) * (Ar[i] - Br[i]);
        }

        return r;
    }

    public static double CalSim(double A[], double B[]) {
        double r = 0.0D;
        int fNum = A.length;
        for (int i = 0; i < fNum; i++) {
            r += (A[i] - B[i]) * (A[i] - B[i]);
        }

        return r;
    }

    public static double CalMean(double data[]) {
        double mean = 0.0D;
        for (int i = 0; i < data.length; i++) {
            mean += data[i];
        }

        return mean / (double) data.length;
    }

    public static double CalErr(Instances data) {
        double m[] = new double[data.numAttributes() - 1];
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            m[i] = 0.0D;
        }

        for (int i = 0; i < data.numAttributes() - 1; i++) {
            m[i] = CalMean(data.attributeToDoubleArray(i));
        }

        double err = 0.0D;
        for (int i = 0; i < data.numInstances(); i++) {
            for (int j = 0; j < data.numAttributes() - 1; j++) {
                err += (data.instance(i).value(j) - m[j]) * (data.instance(i).value(j) - m[j]);
            }

        }

        return err;
    }

    private static double calTPR(double P[], double T[]) {
        double num = 0.0D;
        for (int i = 0; i < P.length; i++) {
            if (T[i] == 0.0D && T[i] == P[i]) {
                num++;
            }
        }

        return num / (double) P.length;
    }

    private static double calFPR(double P[], double T[]) {
        double num = 0.0D;
        for (int i = 0; i < P.length; i++) {
            if (T[i] == 1.0D && T[i] != P[i]) {
                num++;
            }
        }

        return num / (double) P.length;
    }

    public static double calAUC(double Conf[], double T[]) {
        double tmpConf[] = new double[Conf.length];
        double TP[] = new double[Conf.length];
        double FP[] = new double[Conf.length];
        for (int i = 0; i < tmpConf.length; i++) {
            tmpConf[i] = Conf[i];
        }

        int pt[] = sort_index(tmpConf);
        for (int i = 0; i < tmpConf.length; i++) {
            double P[] = new double[tmpConf.length];
            for (int j = 0; j < tmpConf.length; j++) {
                if (j < i) {
                    P[pt[j]] = 0.0D;
                } else {
                    P[pt[j]] = 1.0D;
                }
            }

            TP[i] = calTPR(P, T);
            FP[i] = calFPR(P, T);
        }

        double auc = 0.0D;
        for (int i = 1; i < tmpConf.length; i++) {
            auc += (FP[i] - FP[i - 1]) * (TP[i] + TP[i - 1]);
        }

        return 1.0D - auc / 2D;
    }
}
