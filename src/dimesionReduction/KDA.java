package dimesionReduction;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import util.MyMath;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.EigenvalueDecomposition;
import weka.core.matrix.Matrix;

public class KDA {

    private Instances srcData;
    private Matrix kernelM;
    private Matrix W;
    Matrix sumK;
    private int[] num;
    private Matrix eigVec;
    private Matrix eigVal;
    private int rankKM;
    private Matrix betaVec;
    private Matrix betaVal;
    private Matrix aphal;
    private double omit;

    public KDA(Instances data) {
        data.sort(data.numAttributes() - 1);
        this.srcData = new Instances(data);
        this.num = new int[data.numClasses()];
        this.kernelM = new Matrix(data.numInstances(), data.numInstances());
        this.W = new Matrix(data.numInstances(), data.numInstances());
    }

    private void buildKM() {
        this.sumK = new Matrix(this.srcData.numInstances(), this.srcData.numInstances());
        Matrix ones = new Matrix(this.srcData.numInstances(), this.srcData.numInstances());
        this.omit = 4.9E-324D;
        MyMath mh = new MyMath();
        for (int i = 0; i < this.srcData.numInstances(); i++) {
            for (int j = 0; j < this.srcData.numInstances(); j++) {
                double v = MyMath.CalSim(this.srcData.instance(i), this.srcData.instance(j));
                if (this.srcData.instance(i).classValue() != this.srcData.instance(j).classValue()) {
                    this.omit = Math.max(v, this.omit);
                }
            }
        }
        this.omit = this.omit;

        for (int i = 0; i < this.srcData.numInstances(); i++) {
            for (int j = 0; j < this.srcData.numInstances(); j++) {
                double val = Math.exp(-MyMath.CalSim(this.srcData.instance(i), this.srcData.instance(j)) / this.omit);
                this.kernelM.set(i, j, val);
                ones.set(i, j, 1.0D / this.srcData.numInstances());
            }
        }
        this.sumK = ones.times(this.kernelM);

        this.kernelM = this.kernelM.minus(this.sumK.transpose()).minus(this.sumK).plus(this.sumK.times(ones));
    }

    private void buildW() {
        int pt = 0;
        int i = 0;
        for (i = 0; i < this.srcData.numInstances(); i++) {
            for (int j = 0; j < this.srcData.numInstances(); j++) {
                this.W.set(i, j, 0.0D);
            }
        }
        for (i = 0; i < this.srcData.numClasses(); i++) {
            for (int j = 0; j < this.num[i]; j++) {
                for (int k = 0; k < this.num[i]; k++) {
                    this.W.set(pt + j, pt + k, 1.0D / this.num[i]);
                }
            }
            pt += this.num[i];
        }
    }

    private void calEig() {
        EigenvalueDecomposition ED = this.kernelM.eig();
        this.eigVal = ED.getD();
        this.eigVec = ED.getV();
        new MyMath();
        MyMath.quickSort(this.eigVal, this.eigVec);
        this.rankKM = 0;
        double minV = this.eigVal.get(0, 0) / 1000.0D;
        for (int i = 0; i < this.eigVal.getRowDimension(); i++) {
            if (this.eigVal.get(i, i) < minV) {
                break;
            }
            this.rankKM += 1;
        }
        this.rankKM = Math.min(this.rankKM, this.srcData.numInstances());
    }

    private void comNewKennel() {
        this.eigVal = this.eigVal.getMatrix(0, this.rankKM - 1, 0, this.rankKM - 1);
        this.eigVec = this.eigVec.getMatrix(0, this.eigVec.getRowDimension() - 1, 0, this.rankKM - 1);
        this.kernelM = this.eigVec.times(this.eigVal).times(this.eigVec.transpose());
    }

    private void calSpNum() {
        int i = 0;
        for (i = 0; i < this.srcData.numClasses(); i++) {
            this.num[i] = 0;
        }
        for (i = 0; i < this.srcData.numInstances(); i++) {
            this.num[((int) this.srcData.instance(i).classValue())] += 1;
        }
    }

    private void calBeta() {
        Matrix newT = this.eigVec.transpose().times(this.W).times(this.eigVec);
        EigenvalueDecomposition ED = newT.eig();
        this.betaVal = ED.getD();
        this.betaVec = ED.getV();
        new MyMath();
        MyMath.quickSort(this.betaVal, this.betaVec);
    }

    private void calAphal() {
        Matrix tmp = this.eigVec.times(this.eigVal.inverse()).times(this.betaVec);
        this.aphal = new Matrix(this.srcData.numInstances(), this.rankKM);
        for (int i = 0; i < this.rankKM; i++) {
            Matrix tv = tmp.getMatrix(0, this.srcData.numInstances() - 1, i, i);
            double val = Math.sqrt(tv.transpose().times(this.kernelM).times(tv).get(0, 0));
            for (int j = 0; j < this.srcData.numInstances(); j++) {
                this.aphal.set(j, i, tmp.get(j, i) / val);
            }
        }
    }

    public Instances decInstances(Instances newData) {
        MyMath mh = new MyMath();
        int i = 0, j = 0;
        Matrix newKernel = new Matrix(newData.numInstances(), this.srcData.numInstances());
        for (i = 0; i < newData.numInstances(); i++) {
            double sum = 0.0D;
            for (j = 0; j < this.srcData.numInstances(); j++) {
                double val = Math.exp(-MyMath.CalSim(newData.instance(i), this.srcData.instance(j)) / this.omit);
                newKernel.set(i, j, val);
                sum += val;
            }
            for (j = 0; j < this.srcData.numInstances(); j++) {
                newKernel.set(i, j, newKernel.get(i, j) - sum / this.srcData.numInstances());
            }
        }
        Matrix newSum = this.sumK.getMatrix(0, 0, 0, this.sumK.getRowDimension() - 1);
        double sumV = 0.0D;
        for (i = 0; i < this.sumK.getRowDimension(); i++) {
            sumV += newSum.get(0, i);
        }
        for (i = 0; i < this.sumK.getRowDimension(); i++) {
            newSum.set(0, i, newSum.get(0, i) - sumV / this.srcData.numInstances());
        }
        for (i = 0; i < newData.numInstances(); i++) {
            for (j = 0; j < this.srcData.numInstances(); j++) {
                newKernel.set(i, j, newKernel.get(i, j) - newSum.get(0, j));
            }
        }
        newKernel = newKernel.times(this.aphal);

        FastVector fv = new FastVector();

        for (i = 0; i < newKernel.getColumnDimension(); i++) {
            fv.addElement(new Attribute("F" + i));
        }

        FastVector classAttribute = new FastVector();
        for (i = 0; i < newData.numClasses(); i++) {
            classAttribute.addElement(i);
        }
        fv.addElement(new Attribute("class", classAttribute));

        Instances disData = new Instances("RD_" + this.srcData.relationName(), fv, newData.numInstances());
        for (i = 0; i < newKernel.getRowDimension(); i++) {
            Instance f = new Instance(newKernel.getColumnDimension() + 1);
            for (j = 0; j < newKernel.getColumnDimension(); j++) {
                f.setValue(j, newKernel.get(i, j));
            }
            f.setValue(j, (int) newData.instance(i).classValue());
            disData.add(f);
        }
        disData.setClassIndex(newKernel.getColumnDimension());
        return disData;
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

    public static void main(String[] args) throws Exception {
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
}
