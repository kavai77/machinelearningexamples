import weka.core.matrix.Matrix;

import java.util.Scanner;

public class HousePricePredictionWithNormalEquation {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int n = scanner.nextInt(); // number of examples
        int m = scanner.nextInt(); // number of features
        Matrix X = new Matrix(m, n+ 1);
        Matrix Y = new Matrix(m, 1);
        readTrainingSet(scanner, n, m, X, Y);

        // training - normal form
        Matrix XT = X.transpose();
        Matrix theta = XT.times(X).inverse().times(XT).times(Y);

        Matrix T = readLiveData(scanner, n);

        // prediction
        final Matrix predictions = T.times(theta);
        
        System.out.println(predictions);
    }

    private static void readTrainingSet(Scanner scanner, int n, int m, Matrix x, Matrix y) {
        for(int i = 0; i < m; i++) {
            x.set(i, 0, 1);
            for(int j = 0; j < n; j++) {
                x.set(i, j + 1, scanner.nextDouble());
            }
            y.set(i, 0, scanner.nextDouble());
        }
    }

    private static Matrix readLiveData(Scanner scanner, int n) {
        int t = scanner.nextInt();
        Matrix T = new Matrix(t, n + 1);
        for(int i = 0; i < t; i++) {
            T.set(i, 0, 1);
            for (int j = 0; j < n; j++) {
                T.set(i, j + 1, scanner.nextDouble());
            }
        }
        return T;
    }
}
