package edu.neu.main;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import Jama.Matrix;

public class Logistic {

	/** the learning rate */
	private double rate;

	/** the weight to learn */
	private double[] weights;

	/** the number of iterations */
	private int ITERATIONS = 6000;

	public Logistic(int n) {
		this.rate = 0.001;
		weights = new double[n];
	}

	private double sigmoid(double z) {
		return 1.0 / (1 + Math.exp(-z));
	}

	public void train(List<Instance> instances) {

		weights = new double[instances.get(0).getDimension()];
		for (int n = 0; n < ITERATIONS; n++) {
			double lik = 0.0;
			int rowLength = instances.size();
			int colLength = instances.get(0).getDimension();
			double[] predictVec = new double[rowLength];
			double[] actualVec = new double[rowLength];
			double[] firstDerivative = new double[colLength];

			for (int i = 0; i < rowLength; i++) {
				double[] x = instances.get(i).getX();
				actualVec[i] = instances.get(i).getLabel();
				double predicted = classify(x);
				predictVec[i] = predicted;
			}

			// update weights and bias
			/**************** Please Fill Missing Lines Here *****************/

			// calculating the first derivative of the cost function
			for (int i = 0; i < rowLength; i++) {
				double error = actualVec[i] - predictVec[i];
				for (int j = 0; j < colLength; j++) {
					firstDerivative[j] += ((instances.get(i).getX()[j] * error));
				}
			}

			Matrix secDer = new Matrix(colLength, colLength);

			for (int i = 0; i < rowLength; i++) {
				double predict = predictVec[i];
				double multiplier = predict * (1 - predict);
				Matrix x = new Matrix(new double[][] { instances.get(i).getX() });
				secDer.plusEquals(x.transpose().times(x).times(multiplier));
			}

			secDer.timesEquals(-1);

			// L2 regularization
			double lambda = 0.1;
			secDer.minusEquals((Matrix.identity(7, 7).times(lambda)));
			Matrix firstDer = new Matrix(firstDerivative,
					firstDerivative.length);

			Matrix inverseSecDer = secDer.inverse();

			Matrix delta = inverseSecDer.times(firstDer);
			Matrix weightMat = new Matrix(weights, weights.length);

			weightMat.minusEquals(delta);
			weights = weightMat.getColumnPackedCopy();

			// calculate log likelihood function
			for (int i = 0; i < rowLength; i++) {
				double[] x = instances.get(i).getX();	
				Matrix xMat = new Matrix(x,x.length);
				xMat = xMat.transpose();
				Matrix mul = xMat.times(weightMat);
				double scalar = mul.get(0, 0);
				lik += ((actualVec[i] * scalar) - Math.log(1 + Math.exp(scalar)));
			}
			System.out.println("iteration: " + n + " " + Arrays.toString(weights) + " mle: " + (- (lik/rowLength)) );
		}
	}

	private double classify(double[] x) {
		double logit = 0;
		for (int i = 0; i < weights.length; i++) {
			logit += weights[i] * x[i];
		}
		return sigmoid(logit);
	}

	public static void main(String... args) throws FileNotFoundException {
		List<Instance> instances = DataSet.readDataSet("data.txt");
		Logistic logistic = new Logistic(instances.get(0).getDimension());
		int kFold = 5;
		double totalAccuracy = 0;
		Collections.shuffle(instances);
		for (int i = 0; i < kFold; i++) {

			List<Instance> train = new ArrayList<>();
			List<Instance> test = new ArrayList<>();

			int foldLB = (instances.size() / kFold) * (i);
			int foldUB = (instances.size() / kFold) * (i + 1);

			for (int j = 0; j < instances.size(); j++) {
				if (foldLB <= j && j < foldUB) {
					test.add(instances.get(j));
				} else {
					train.add(instances.get(j));
				}
			}

			logistic.train(train);
			double[] actualVec = new double[test.size()];

			double correctPrediction = 0;
			for (int k = 0; k < test.size(); k++) {
				double[] x = instances.get(k).getX();
				actualVec[i] = instances.get(k).getLabel();

				double predicted = logistic.classify(x);

				if ((instances.get(k).getLabel() == 1 && predicted >= 0.5)
						|| (instances.get(k).getLabel() == 0 && predicted < 0.5)) {
					correctPrediction++;
				}
			}
			totalAccuracy += (correctPrediction / test.size());
		}
		System.out.println("Average Accuracy is : " + (totalAccuracy/kFold)*100 + "%");
		// toy test
		double[] testPoint = { 1, 63.0278175, 22.55258597, 39.60911701,40.47523153, 98.67291675, -0.254399986 };
		System.out.println("prob(1|testPoint) = " + logistic.classify(testPoint));

	}
}
