// Decompiled by Jad v1.5.8g. Copyright 2001 Pavel Kouznetsov.
// Jad home page: http://www.kpdus.com/jad.html
// Decompiler options: packimports(3) 
// Source File Name:   BRSD_BK.java
package zwj.util;

import java.io.*;

import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;

public class BRSD_BK {

    public BRSD_BK(Instances data, int trueL[]) {
        src = new Instances(data);
        label = trueL;
        clusterLabel = new int[src.numInstances()];

        numCluster = 1;
    }

    public int[] getLable() {
        return clusterLabel;
    }

    public void buildCluster()
            throws Exception {
        int Q[] = new int[src.numInstances() + 1];
        int lNum = label.length;
        int had[] = new int[src.numInstances() + 1];
        int r;
        int f = r = 0;
        Q[f++] = 0;
        had[0] = 1;
        while (f != r) {
            Instances now = new Instances(src, 0);
            Instances nowL = new Instances(src, 0);
            int pt = Q[r++];
            int tmpPt = 0;
            for (int i = 0; i < src.numInstances(); i++) {
                if (clusterLabel[i] == pt) {
                    now.add((Instance) src.instance(i).copy());
                    if (i < lNum) {
                        nowL.add((Instance) src.instance(i).copy());
                    }
                }
            }

            int tmpL[] = new int[lNum];
            for (int i = 0; i < lNum; i++) {
                if (clusterLabel[i] == pt) {
                    tmpL[tmpPt++] = label[i];
                }
            }

            SimpleKMeans km = new SimpleKMeans();
            km.setNumClusters(2);
            km.buildClusterer(now);
            Instances A = new Instances(now, 0);
            Instances B = new Instances(now, 0);
            int a = 0;
            int b = 0;
            for (int i = 0; i < src.numInstances(); i++) {
                if (clusterLabel[i] == pt) {
                    if (km.clusterInstance(src.instance(i)) == 0) {
                        A.add((Instance) src.instance(i).copy());
                        if (i < lNum) {
                            a++;
                        }
                    } else {
                        B.add((Instance) src.instance(i).copy());
                        if (i < lNum) {
                            b++;
                        }
                    }
                }
            }

            if ((a != 0 || b != 0) && A.numInstances() > 1 && B.numInstances() > 1 && (calPur(nowL, tmpL) < 0.90000000000000002D || calPar(now, A, B))) {
                int spt = -1;
                for (int i = 0; i < src.numInstances(); i++) {
                    if (had[i] == 1) {
                        continue;
                    }
                    spt = i;
                    had[i] = 1;
                    break;
                }

                Q[f++] = pt;
                Q[f++] = spt;
                for (int i = 0; i < src.numInstances(); i++) {
                    if (clusterLabel[i] == pt) {
                        if (km.clusterInstance(src.instance(i)) == 0) {
                            clusterLabel[i] = pt;
                        } else {
                            clusterLabel[i] = spt;
                        }
                    }
                }

                numCluster++;
            }
        }
        int flag[] = new int[src.numInstances() + 1];
        int nowpt = 0;
        for (int i = 0; i < src.numInstances(); i++) {
            flag[i] = -1;
        }

        for (int i = 0; i < src.numInstances(); i++) {
            if (flag[clusterLabel[i]] == -1) {
                flag[clusterLabel[i]] = nowpt++;
            }
            clusterLabel[i] = flag[clusterLabel[i]];
        }

    }

    private double calPur(Instances data, int trueL[]) {
        double pur = 0.0D;
        double numA = 0.0D;
        double numB = 0.0D;
        for (int i = 0; i < data.numInstances(); i++) {
            if (trueL[i] == 0) {
                numA++;
            }
            if (trueL[i] == 1) {
                numB++;
            }
        }

        if (numB > numA) {
            pur = numB / (double) data.numInstances();
        } else {
            pur = numA / (double) data.numInstances();
        }
        return pur;
    }

    private boolean calPar(Instances All, Instances A, Instances B) {
        boolean flag = false;
        double tt = calErr(All);
        double tA = calErr(A);
        double tB = calErr(B);
        if (tt > tA + tB) {
            flag = true;
        }
        return flag;
    }

    private double calErr(Instances data) {
        double m[] = new double[data.numAttributes() - 1];
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            m[i] = 0.0D;
        }

        for (int i = 0; i < data.numAttributes() - 1; i++) {
            for (int j = 0; j < data.numInstances(); j++) {
                m[i] += data.instance(j).value(i);
            }

            m[i] /= data.numInstances();
        }

        double err = 0.0D;
        for (int i = 0; i < data.numInstances(); i++) {
            for (int j = 0; j < data.numAttributes() - 1; j++) {
                err += (data.instance(i).value(j) - m[j]) * (data.instance(i).value(j) - m[j]);
            }

        }

        return err;
    }

    public static void main(String args[])
            throws Exception {
        Instances Data = new Instances(new BufferedReader(new FileReader("DataSet/landmine dataset/Task21.arff")));
        Data.setClassIndex(-1);
        int label[] = new int[Data.numInstances()];
        for (int i = 0; i < Data.numInstances(); i++) {
            label[i] = (int) Data.instance(i).value(Data.numAttributes() - 1);
        }

        Data.deleteAttributeAt(Data.numAttributes() - 1);
        BRSD_BK bk = new BRSD_BK(Data, label);
        bk.buildCluster();
        label = bk.getLable();
        FileWriter labelFile = new FileWriter("ans.dat", true);
        for (int i = 0; i < label.length; i++) {
            labelFile.write((new StringBuilder(String.valueOf(Data.instance(i).value(0)))).append(" ").append(Data.instance(i).value(1)).append(" ").append(label[i]).append("\n").toString());
        }

        labelFile.close();
    }

    private Instances src;
    private int label[];
    private int clusterLabel[];
    private int numCluster;
}
