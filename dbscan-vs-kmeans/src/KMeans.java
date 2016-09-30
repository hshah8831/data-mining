package edu.neu.main;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import Jama.Matrix;

public class KMeans {
	private Matrix data;
	private Matrix clusterCenter;
	private Matrix weight;
	private Matrix finalOutput;

	public Matrix getData() {
		return data;
	}

	public void setData(Matrix data) {
		this.data = data;
	}

	public Matrix getClusterCenter() {
		return clusterCenter;
	}

	public void setClusterCenter(Matrix clusterCenter) {
		this.clusterCenter = clusterCenter;
	}

	public Matrix getWeight() {
		return weight;
	}

	public void setWeight(Matrix weight) {
		this.weight = weight;
	}

	/**
	 * reads from the filename into the data matrix
	 */
	public KMeans(String fileName, int k) {
		List<double[]> listData = new ArrayList<>();
		String line;
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(fileName));
			while ((line = br.readLine()) != null) {
				String rowData[] = line.split("\t");
				double rowDoubleData[] = new double[rowData.length];
				// To convert String data into double
				for (int i = 0; i < rowData.length; ++i) {
					rowDoubleData[i] = Double.parseDouble(rowData[i]);
				}
				listData.add(rowDoubleData);
			}
		} catch (NumberFormatException | IOException e) {
			e.printStackTrace();
		}
		int data_cols = listData.get(0).length;
		int data_rows = listData.size();

		Matrix matrixData = new Matrix(data_rows, data_cols);
		// storing data in Matrix
		for (int r = 0; r < data_rows; r++) {
			for (int c = 0; c < data_cols; c++) {
				matrixData.set(r, c, listData.get(r)[c]);
			}
		}
		data = matrixData;
		int[] randomInd = new int[k];
		Random rand = new Random();
		int randomNum = rand.nextInt(matrixData.getRowDimension());
		for (int i = 0; i < k; i++) {
			randomInd[i] = randomNum + i;
		}
		clusterCenter = matrixData.getMatrix(randomInd, 0, data_cols - 2);
//		clusterCenter = new Matrix(k, data_cols - 1);
//		clusterCenter.print(6, 2);
	}

	public void populateWeightMatrix() {
		double min = 9999999999.0d;
		int center = 0;
		Matrix weightData = new Matrix(data.getRowDimension(), clusterCenter.getRowDimension());
		for (int i = 0; i < data.getRowDimension(); i++) {
			Matrix dataRow = data.getMatrix(i, i, 0, data.getColumnDimension() - 2);
			double distance = min;
			center = 0;
			min = 9999999999.0d;
			for (int j = 0; j < clusterCenter.getRowDimension(); j++) {
				distance = getDistance(dataRow, clusterCenter.getMatrix(j, j, 0, clusterCenter.getColumnDimension() - 1));
				if (distance < min) {
					center = j;
					min = distance;
				}
			}
			weightData.set(i, center, 1);
		}
		weight = weightData;
	}

	private double getDistance(Matrix dataRow, Matrix centerRow) {
		double distance = 0.0d;
		for (int i = 0; i < dataRow.getColumnDimension(); i++) {
			distance += (Math.pow(dataRow.get(0, i) - centerRow.get(0, i), 2));
		}
		return distance;
	}

	public void updateCenter() {
		for (int i = 0; i < weight.getColumnDimension(); i++) {
			int count = 0;
			Matrix sum = new Matrix(1, data.getColumnDimension() - 1);
			for (int j = 0; j < data.getRowDimension(); j++) {
				if (weight.get(j, i) == 1) {
					count++;
					sum.plusEquals(data.getMatrix(j, j, 0, data.getColumnDimension() - 2));
				}
			}
			sum.timesEquals((double)1 / count);
			clusterCenter.setMatrix(i, i, 0, sum.getColumnDimension() - 1, sum);
		}
	}

	public void printWeight() {
		System.out.println("Centre are");
		clusterCenter.print(10, 3);
		Matrix outputMatrix = (Matrix) data.clone();
		for(int i = 0; i < outputMatrix.getRowDimension(); i++){
			for(int j = 0; j < weight.getColumnDimension(); j++){
				if(weight.get(i, j) == 1.0d){
					outputMatrix.set(i, outputMatrix.getColumnDimension() - 1, j + 1);
				}
			}
		}
		finalOutput = outputMatrix;
		outputMatrix.print(6, 2);
		FileWriter fStream;
		try {
			fStream = new FileWriter("output\\temp.txt");
			PrintWriter out = new PrintWriter(fStream);
			outputMatrix.print(out, 6, 2);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void getPurity(){
		int[][] purityMat = new int[weight.getColumnDimension()][weight.getColumnDimension()];
		int[] groundTruthArr = new int[weight.getColumnDimension()];
		int[] clusterArr = new int[weight.getColumnDimension()];
		for(int i = 0; i < data.getRowDimension(); i++){
			int groundTruth = (int) data.get(i, data.getColumnDimension() - 1);
			int cluster = (int) finalOutput.get(i, data.getColumnDimension() - 1);
			purityMat[cluster - 1][groundTruth - 1]++;
			groundTruthArr[groundTruth - 1]++;
			clusterArr[cluster - 1]++;
		}
		double purity = 0.0d;
		for(int i = 0; i < weight.getColumnDimension(); i++){
			double max = purityMat[i][0];
			for(int j = 1; j < weight.getColumnDimension(); j++){
				if(purityMat[i][j] > max){
					max = purityMat[i][j];
				}
			}
			purity += max;
		}
		int N = data.getRowDimension();
		double information = 0.0d;
		double entropyOmega = 0.0d;
		double entropyCluster = 0.0d;
		double NMI = 0.0d;
		for(int i = 0; i < weight.getColumnDimension(); i++){
			for(int j = 0; j < weight.getColumnDimension(); j++){
				if(purityMat[i][j] != 0){
					information += ((double)purityMat[i][j]/N) * Math.log((double)(N * purityMat[i][j])/(clusterArr[i] * groundTruthArr[j]));
				}
			}
		}
		
		for(int i = 0; i < groundTruthArr.length; i++){
			entropyCluster -= ((double)groundTruthArr[i]/N) * Math.log((double)groundTruthArr[i]/N); 
		}
		
		for(int i = 0; i < clusterArr.length; i++){
			entropyOmega -= ((double)clusterArr[i]/N) * Math.log((double)clusterArr[i]/N); 
		}
		NMI = information/Math.sqrt(entropyOmega * entropyCluster);
		System.out.println("Purity = " + purity/N + " NMI = " + NMI);
	}

	public static boolean isCenterEqual(Matrix newCenter, Matrix oldCenter){
		boolean ret = true;
		int i = 0,j = 0;
		for(i = 0; i < oldCenter.getRowDimension() && ret == true; i++){
			for(j = 0; j < oldCenter.getColumnDimension(); j++){
				if(oldCenter.get(i, j) != newCenter.get(i, j)){
					ret = false;break;
				}
			}
		}
		return ret;
	}
	
	public static void main(String[] args) {
		KMeans kmeans = new KMeans("input\\dataset3.txt", 2);
		Matrix oldCenters = kmeans.getClusterCenter();
		Matrix newCenters = new Matrix(oldCenters.getRowDimension(),oldCenters.getColumnDimension());
		while (!isCenterEqual(newCenters, oldCenters)) {
			kmeans.populateWeightMatrix();
			oldCenters = (Matrix)kmeans.getClusterCenter().clone();
			kmeans.updateCenter();
			newCenters = (Matrix)kmeans.getClusterCenter().clone();
		}
		kmeans.printWeight();
		kmeans.getPurity();

	}

}
