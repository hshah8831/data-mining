package edu.neu.main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import Jama.Matrix;

public class DBScan {
	private Matrix data;
	private double epsilon;
	private Map<Integer, List<Integer>> epsaHood;
	private List<ArrayList<Integer>> clusters;
	private Matrix outputData;

	public DBScan(String fileName, int k) {
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
		epsilon = 0.0d;
		for (int i = 0; i < data.getRowDimension(); i++) {
			double[] kNearestDist = { 999999.0d, 999999.0d, 999999.0d };
			for (int j = 0; j < data.getRowDimension(); j++) {
				if (i != j) {
					double dist = getDistance(
							data.getMatrix(i, i, 0, data_cols - 2),
							data.getMatrix(j, j, 0, data_cols - 2));
					kNearestDist = update(kNearestDist, dist);
				}
			}
			epsilon += largestOf(kNearestDist);
		}
		epsilon /= data_rows;
		epsaHood = createEpsaHoodMap();
		clusterData();
		printWeight();
	}

	private double[] update(double[] kNearestDist, double dist) {
		for (int i = 0; i < kNearestDist.length; i++) {
			if (dist < kNearestDist[i]) {
				kNearestDist[i] = dist;
				break;
			}
		}
		return kNearestDist;
	}

	private double largestOf(double[] kNearestDist) {
		double max = kNearestDist[0];
		for (int i = 1; i < kNearestDist.length; i++) {
			if (kNearestDist[i] > max) {
				max = kNearestDist[i];
			}
		}
		return max;
	}

	private double getDistance(Matrix dataRow, Matrix centerRow) {
		double distance = 0.0d;
		for (int i = 0; i < dataRow.getColumnDimension(); i++) {
			distance += (Math.pow(dataRow.get(0, i) - centerRow.get(0, i), 2));
		}
		return Math.sqrt(distance);
	}

	private Map<Integer, List<Integer>> createEpsaHoodMap() {
		Map<Integer, List<Integer>> map = new HashMap<Integer, List<Integer>>();
		for (int i = 0; i < data.getRowDimension(); i++) {
			List<Integer> index = new ArrayList<Integer>();
			for (int j = 0; j < data.getRowDimension(); j++) {
				if (i != j) {
					double dist = getDistance(data.getMatrix(i, i, 0,
							data.getColumnDimension() - 2), data.getMatrix(j,
							j, 0, data.getColumnDimension() - 2));
					if (dist < epsilon) {
						index.add(j);
					}
				}
			}
			map.put(i, index);
		}
		return map;
	}

	public void clusterData() {
		Set<Integer> visited = new HashSet<Integer>();
		List<ArrayList<Integer>> localClusters = new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> noise = new ArrayList<Integer>();
		while (visited.size() < data.getRowDimension()) {
			Random rand = new Random();
			int index = rand.nextInt(data.getRowDimension());
			if (!visited.contains(index)) {
				visited.add(index);
				if (epsaHood.get(index).size() >= 3) {
					List<Integer> newCluster = new ArrayList<Integer>();
					newCluster.add(index);
					Deque<Integer> neighbour = new LinkedList<Integer>(
							epsaHood.get(index));
					int in = 0;
					while (!neighbour.isEmpty()) {
						in = neighbour.pollFirst();
						if (!visited.contains(in)) {
							visited.add(in);
							if (epsaHood.get(in).size() >= 3) {
								neighbour.addAll(epsaHood.get(in));
							}
							if (!assignedAny(localClusters, in)) {
								newCluster.add(in);
							}
						}
					}
					localClusters.add((ArrayList<Integer>) newCluster);
				} else {
					noise.add(index);
				}
			}
		}
		localClusters.add(noise);
		clusters = localClusters;
	}

	public void printWeight() {
		Matrix outputMatrix = (Matrix) data.clone();
		int n = 0;
		for (ArrayList<Integer> cluster : clusters) {
			n++;
			for (Integer index : cluster) {
				outputMatrix.set(index, outputMatrix.getColumnDimension() - 1,
						n);
			}
		}
		outputData = outputMatrix;
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

	private boolean assignedAny(List<ArrayList<Integer>> clusters, Integer index) {
		boolean ret = false;
		for (ArrayList<Integer> cluster : clusters) {
			if (cluster.contains(index))
				ret = true;
		}
		return ret;
	}
	
	public void getPurity(){
		int actualLabels = 2;
		int[][] purityMat = new int[clusters.size()][actualLabels];
		int[] groundTruthArr = new int[actualLabels];
		int[] clusterArr = new int[clusters.size()];
		for(int i = 0; i < data.getRowDimension(); i++){
			int groundTruth = (int) data.get(i, data.getColumnDimension() - 1);
			int cluster = (int) outputData.get(i, data.getColumnDimension() - 1);
			purityMat[cluster - 1][groundTruth - 1]++;
			groundTruthArr[groundTruth - 1]++;
			clusterArr[cluster - 1]++;
		}
		double purity = 0.0d;
		for(int i = 0; i < clusters.size(); i++){
			double max = purityMat[i][0];
			for(int j = 1; j < actualLabels; j++){
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
		for(int i = 0; i < clusters.size(); i++){
			for(int j = 0; j < actualLabels; j++){
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

	public static void main(String[] args) {
		DBScan dbscan = new DBScan("input\\dataset3.txt", 3);
		dbscan.getPurity();
	}
}
