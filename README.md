## 実装している内容
**モデル:**　重回帰

**再帰関数:**　勾配降下法
```
public void train(List<List<Double>> data, double learningRate, int numEpochs, int batchSize, double beta1, double beta2, double epsilon) {
    // ...
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        // ...
        for (int i = 0; i < features.length; i += batchSize) {
            // ...
            for (int j = 0; j < batchFeatures.length; j++) {
                // ...
                t++;
                updateCoefficients(learningRate, beta1, beta2, epsilon, batchFeatures[j], error);
            }
        }
        // ...
    }
}
```
**誤差関数:** Adam
```
private void updateCoefficients(double learningRate, double beta1, double beta2, double epsilon, double[] batchFeatures, double error) {
    // ...
    m[k] = beta1 * m[k] + (1 - beta1) * error * batchFeatures[k];
    v[k] = beta2 * v[k] + (1 - beta2) * Math.pow(error * batchFeatures[k], 2);
    // ...
    coefficients[k] -= learningRate * (mHat + regularizationTerm) / (Math.sqrt(vHat) + epsilon);
}
```
**パラメータ調整:**　正則化
```
public MultipleLinearRegression(int numFeatures, double learningRate, double beta1, double beta2, double epsilon, double lambda) {
    // ...
    this.lambda = lambda;
    initializeCoefficients();
}
```

**その他:** データは正規化+シャッフルしています

正規化
```
private void normalizeFeatures(List<List<Double>> data) {
    // ...
    for (List<Double> row : data) {
        // ...
        for (int i = 0; i < numFeatures + 1; i++) {
            // ...
            double normalizedValue = row.get(i) / featureMaxValues[i - 1];
            row.set(i, normalizedValue);
        }
    }
}
```
