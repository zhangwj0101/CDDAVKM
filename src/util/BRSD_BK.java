package util;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;

public class BRSD_BK {

    private Instances src;
    private int[] label;
    private int[] clusterLabel;
    private int numCluster;

    public BRSD_BK(Instances data, int[] trueL) {
        this.src = new Instances(data);
        this.label = trueL;
        this.clusterLabel = new int[this.src.numInstances()];
        for (int i = 0; i < this.src.numInstances(); i++) {
            this.clusterLabel[i] = 0;
        }
        this.numCluster = 1;
    }

    public int[] getLable() {
        return this.clusterLabel;
    }

    public void buildCluster() throws Exception {
        int[] Q = new int[this.src.numInstances() + 1];

        int lNum = this.label.length;
        int i = 0;
        int[] had = new int[this.src.numInstances() + 1];
        int r;
        int f = r = 0;
        Q[(f++)] = 0;
        had[0] = 1;
        while (f != r) {
            Instances now = new Instances(this.src, 0);
            Instances nowL = new Instances(this.src, 0);

            int pt = Q[(r++)];
            int tmpPt = 0;
            for (i = 0; i < this.src.numInstances(); i++) {
                if (this.clusterLabel[i] == pt) {
                    now.add((Instance) this.src.instance(i).copy());
                    if (i < lNum) {
                        nowL.add((Instance) this.src.instance(i).copy());
                    }
                }
            }

            int[] tmpL = new int[lNum];
            for (i = 0; i < lNum; i++) {
                if (this.clusterLabel[i] == pt) {
                    tmpL[(tmpPt++)] = this.label[i];
                }
            }

            SimpleKMeans km = new SimpleKMeans();
            km.setNumClusters(2);
            km.buildClusterer(now);

            Instances A = new Instances(now, 0);
            Instances B = new Instances(now, 0);

            int a = 0;
            int b = 0;

            for (i = 0; i < this.src.numInstances(); i++) {
                if (this.clusterLabel[i] == pt) {
                    if (km.clusterInstance(this.src.instance(i)) == 0) {
                        A.add((Instance) this.src.instance(i).copy());
                        if (i < lNum) {
                            a++;
                        }
                    } else {
                        B.add((Instance) this.src.instance(i).copy());
                        if (i < lNum) {
                            b++;
                        }
                    }
                }
            }

            if ((a != 0) || (b != 0)) {
                if ((A.numInstances() > 1) && (B.numInstances() > 1) && ((calPur(nowL, tmpL) < 0.9D) || (calPar(now, A, B)))) {
                    int spt = -1;
                    for (i = 0; i < this.src.numInstances(); i++) {
                        if (had[i] != 1) {
                            spt = i;
                            had[i] = 1;
                            break;
                        }
                    }

                    Q[(f++)] = pt;
                    Q[(f++)] = spt;
                    for (i = 0; i < this.src.numInstances(); i++) {
                        if (this.clusterLabel[i] == pt) {
                            if (km.clusterInstance(this.src.instance(i)) == 0) {
                                this.clusterLabel[i] = pt;
                            } else {
                                this.clusterLabel[i] = spt;
                            }
                        }
                    }
                    this.numCluster += 1;
                }
            }
        }
        int[] flag = new int[this.src.numInstances() + 1];
        int nowpt = 0;
        for (i = 0; i < this.src.numInstances(); i++) {
            flag[i] = -1;
        }
        for (i = 0; i < this.src.numInstances(); i++) {
            if (flag[this.clusterLabel[i]] == -1) {
                flag[this.clusterLabel[i]] = (nowpt++);
            }
            this.clusterLabel[i] = flag[this.clusterLabel[i]];
        }
    }

    private double calPur(Instances data, int[] trueL) {
        double pur = 0.0D;
        double numA = 0.0D;
        double numB = 0.0D;
        for (int i = 0; i < data.numInstances(); i++) {
            if (trueL[i] == 0) {
                numA += 1.0D;
            }
            if (trueL[i] == 1) {
                numB += 1.0D;
            }
        }
        if (numB > numA) {
            pur = numB / data.numInstances();
        } else {
            pur = numA / data.numInstances();
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
        double[] m = new double[data.numAttributes() - 1];
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

    public static void main(String[] args) throws Exception {
        Instances Data = new Instances(new BufferedReader(new FileReader("DataSet/landmine dataset/Task21.arff")));

        Data.setClassIndex(-1);
        int[] label = new int[Data.numInstances()];
        for (int i = 0; i < Data.numInstances(); i++) {
            label[i] = ((int) Data.instance(i).value(Data.numAttributes() - 1));
        }
        Data.deleteAttributeAt(Data.numAttributes() - 1);
        BRSD_BK bk = new BRSD_BK(Data, label);
        bk.buildCluster();
        label = bk.getLable();
        FileWriter labelFile = new FileWriter("ans.dat", true);
        for (int i = 0; i < label.length; i++) {
            labelFile.write(Data.instance(i).value(0) + " " + Data.instance(i).value(1) + " " + label[i] + "\n");
        }
        labelFile.close();
    }
}
