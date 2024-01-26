import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MultipleLinearRegression {
    private int numFeatures;
    private double[] coefficients;
    private List<List<Double>> featureMaxValuesList = new ArrayList<>();
    private double[] m; // 1st moment estimate
    private double[] v; // 2nd raw moment estimate
    private int t; // Time step
    private double lambda;

    //コンストラクタ
    public MultipleLinearRegression(int numFeatures, double learningRate, double beta1, double beta2, double epsilon, double lambda) {
        this.numFeatures = numFeatures;
        this.coefficients = new double[numFeatures];
        this.m = new double[numFeatures];
        this.v = new double[numFeatures];
        this.t = 0;
        this.lambda = lambda;
        initializeCoefficients();
    }

    // 重み初期化
    private void initializeCoefficients() {
        for (int i = 0; i < numFeatures; i++) {
            coefficients[i] = 0.0; 
        }
    }

    // 正規化
    private void normalizeFeatures(List<List<Double>> data) {
        int numSamples = data.size();
        System.out.println(data.size() + "->Size! ");
        
        double[] featureMaxValues = new double[numFeatures];
        Arrays.fill(featureMaxValues, Double.NEGATIVE_INFINITY);

        // 各特徴量の最大値を計算
        for (List<Double> row : data) {
            for (int i = 0; i < numFeatures; i++) {
                featureMaxValues[i] = Math.max(featureMaxValues[i], row.get(i+1));
                System.out.println(" Max: " + featureMaxValues[i] + " ");
            }
        }

        System.out.print(" 各特徴量の最大値:");
        List<Double> featureMaxValuesList = new ArrayList<>();
        for (int i = 0; i < numFeatures; i++) {
            System.out.print(" Feature " + i + ": " + featureMaxValues[i]);
            
            //0は発散するので極小にする
            if(featureMaxValues[i] == 0){
            	featureMaxValues[i] = 0.0000000001;
        	}
            featureMaxValuesList.add(featureMaxValues[i]);
        }
        this.featureMaxValuesList.add(featureMaxValuesList); // リストに追加

        System.out.println("");

        // 特徴量の正規化
        for (List<Double> row : data) {
            for (int i = 0; i < numFeatures + 1; i++) {
                if (i != 0) {
                    double normalizedValue = row.get(i) / featureMaxValues[i - 1];
                    row.set(i, normalizedValue);
                } else {
                    row.set(0, row.get(0));
                }
            }
        }
    }

    //最大値の計算
    private double[] normalizeFeaturesAndGetMaxValues() {
        double[] featureMaxValue = new double[numFeatures];
        Arrays.fill(featureMaxValue, Double.NEGATIVE_INFINITY);

        for (int i = 0; i < featureMaxValuesList.size(); i++) {
            featureMaxValue[i] = featureMaxValuesList.get(i).get(0);
            System.out.print(" GET-VAL:" + featureMaxValue[i]);
        }

        return featureMaxValue;
    }
    
    // 最大値のgetter
    public double[] getMaxValues() {
        return normalizeFeaturesAndGetMaxValues();
    }

    // 予測
    public double predict(double[] inputs) {
        if (inputs.length != numFeatures) {
            throw new IllegalArgumentException("入力の次元が正しくありません。" + inputs.length + "!=" + numFeatures);
        }

        double prediction = 0.0;
        //System.out.println("入力学習次元数:" + inputs.length);//2
        
        for (int i = 0; i < numFeatures; i++) {
        	//System.out.println(inputs[i]);
            prediction += coefficients[i] * inputs[i];
        }
        
        System.out.println(" 精度:" + sigmoid(prediction));

        return prediction;
    }
    
    //予測時のみ次元が変わるので、回避用のオーバーライド
    public double predict(double[] inputs, boolean mode) {
        double prediction = 0.0;

        if (inputs.length-1 != numFeatures) {
           throw new IllegalArgumentException("入力の次元が正しくありません!" + inputs.length + "!=" + numFeatures);
        }
	    //System.out.println(" 入力学習次元数;" + inputs.length + " zero_num:" + inputs[0]);//3
	        
	    for (int i = 1; i < numFeatures+1; i++) {
	    //System.out.println(inputs[i]);
	       prediction += coefficients[i-1] * inputs[i];
	    }

        return prediction;
        
    }
    
    // シグモイド関数
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // 二乗平均誤差を計算するメソッド
    private double calculateMSE(List<List<Double>> data) {
        double sumSquaredError = 0.0;
        int count = 0;

        // 長さの調節
        // なぜ変わるのかは現在調査中 -> 学習データを読み込んだ後に回答データを抜き出しているから？
        for (List<Double> row : data) {
            if (row.size() == numFeatures) {
                double[] inputs = row.subList(0, numFeatures).stream().mapToDouble(Double::doubleValue).toArray();
                double prediction = predict(inputs);
                double target = row.get(0);
                if (count < 1) {
                    System.out.println("prediction:" + prediction + " target:" + target + " in:" + row);
                }
                double error = prediction - target;
                sumSquaredError += Math.pow(error, 2);

            } else {
                double[] inputs = new double[row.size()];

                for (int i = 0; i < row.size(); i++) {
                    inputs[i] = row.get(i);
                }

                double prediction = predict(inputs, true);
                double target = row.get(0);
                if (count < 1) {
                    System.out.println("prediction:" + prediction + " target:" + target + " in:" + row);
                }
                double error = prediction - target;
                sumSquaredError += Math.pow(error, 2);
            }

            count++;
        }

        double mse = sumSquaredError / data.size();
        double rmse = Math.sqrt(mse);

        // System.out.println(" 精度:" + sigmoid(rmse) + " err:" + rmse + " size:" + data.size());
        return rmse;
    }

    // 最急降下法
    public void train(List<List<Double>> data, double learningRate, int numEpochs, int batchSize, double beta1, double beta2, double epsilon) {
        Random rand = new Random();

        normalizeFeatures(data);

        double[][] features = new double[data.size()][numFeatures + 1];
        double[] targets = new double[data.size()];

        for (int i = 0; i < data.size(); i++) {
            List<Double> row = data.get(i);
            for (int j = 1; j < numFeatures + 1; j++) {
                features[i][j] = row.get(j);
                System.out.println("学習データ：" + features[i][j]);
            }
            targets[i] = row.get(0);
            System.out.println("解データ：" + targets[i]);
        }

        for (int epoch = 0; epoch < numEpochs; epoch++) {
            shuffleData(features, targets, rand);

            for (int i = 0; i < features.length; i += batchSize) {
                double[][] batchFeatures = Arrays.copyOfRange(features, i, Math.min(i + batchSize, features.length));
                double[] batchTargets = Arrays.copyOfRange(targets, i, Math.min(i + batchSize, targets.length));

                // ミニバッチ学習
                for (int j = 0; j < batchFeatures.length; j++) {
                    double prediction = predict(batchFeatures[j], true);
                    double error = prediction - batchTargets[j];

                    // Adam optimizer with L2 regularization
                    t++;
                    for (int k = 0; k < numFeatures; k++) {
                        m[k] = beta1 * m[k] + (1 - beta1) * error * batchFeatures[j][k];
                        v[k] = beta2 * v[k] + (1 - beta2) * Math.pow(error * batchFeatures[j][k], 2);
                        double mHat = m[k] / (1 - Math.pow(beta1, t));
                        double vHat = v[k] / (1 - Math.pow(beta2, t));
                        
                        // L2 regularization term
                        double regularizationTerm = 2 * lambda * coefficients[k];

                        coefficients[k] -= learningRate * (mHat + regularizationTerm) / (Math.sqrt(vHat) + epsilon);
                    }
                }
            }

            double mse = calculateMSE(data);
            System.out.println("epoch: " + epoch + ", RMSE: " + mse);
            System.out.println("");
        }
    }

    // シャッフル
    private void shuffleData(double[][] features, double[] targets, Random rand) {
        int n = features.length;
        for (int i = n - 1; i > 0; i--) {
            int randIndex = rand.nextInt(i + 1);

            // 特徴量
            double[] tempFeatures = features[i];
            features[i] = features[randIndex];
            features[randIndex] = tempFeatures;

            // ターゲット
            double tempTarget = targets[i];
            targets[i] = targets[randIndex];
            targets[randIndex] = tempTarget;
        }
    }

    //テストコード
    public static void main(String[] args) {
    	int numFeatures = 2;
	    double learningRate = 0.001;
	    double beta1 = 0.9;
	    double beta2 = 0.999;
	    double epsilon = 1e-8;
	    double lam = 0.01;
	
	    MultipleLinearRegression regressionModel = new MultipleLinearRegression(numFeatures, learningRate, beta1, beta2, epsilon,lam);
        List<List<Double>> data = new ArrayList<>();
        data.add(Arrays.asList(250.0, 200.0, 10.0)); 
        data.add(Arrays.asList(30.0, 40.0, 0.01)); 

        // 学習
        int numEpochs = 200;
        int batchSize = 2;
        regressionModel.train(data, learningRate, numEpochs, batchSize, beta1, beta2, epsilon);

        // 予測データの設定
        double[] testInputs = {250.0, 200.0, 10.0}; 
        double testTarget = 250.0;  // 解
        
        //ここでの正規化は特殊であるため、個別に行う。->　統合するべき
        for (int i = 1; i < testInputs.length; i++) {
        	System.out.print(" 元値:" + testInputs[i]);
        	testInputs[i] = testInputs[i]/regressionModel.getMaxValues()[i-1];
        	System.out.print("->正規化["+i+"]:"+testInputs[i]);
        	
        	if(testInputs[i] == 0){
        		testInputs[i] = 0.0000000001;
        	}
        }
        System.out.println("");

        // 予測
        double prediction = regressionModel.predict(testInputs,true);
        System.out.println("Predicted: " + prediction * 10000000000000.0);
        System.out.println("Actual Target: " + testTarget);
    }
}
