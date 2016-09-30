package org.neu.main;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ThreadLocalRandom;

import org.neu.main.MatrixData;

import Jama.Matrix;

public class KNN {
	
	/**
	 * Calculates and returns the euclidean distance of the input rows. Invariant is the size of each row MUST be same.
	 * @param row1	
	 * @param row2
	 * @return	the Euclidean distance of the input rows
	 */	
	private static double getEuclideanDistance(double[] row1, double[] row2){
		double dist = 0;
		for(int i = 0; i < row1.length; i++){
			dist += (Math.pow((row1[i] - row2[i]), 2));
		}
		return Math.sqrt(dist);
	}
	
	/**
	 * replaces the biggest index with incoming index and dist if new distance is smaller than the largest distance
	 * @param map map that contains the Index,Distance mapping of the smallest distance seen so far
	 * @param index index of the incoming row
	 * @param dist euclidean distance of the incoming row with the current test row
	 * @param k defines how many nearest neighbours we are looking for
	 * @return the updated index,distance map
	 */
	private static Map<Integer,Double> replaceBiggestIndex(Map<Integer,Double> map, int index, double dist,int k){
		if(map.size()<k){
			map.put(index, dist);
		} else {
			int maxIndex = -1;
			for(Entry<Integer,Double> pair : map.entrySet()){
				if(pair.getValue() > dist){
					maxIndex = pair.getKey();
				}
			}
			if(maxIndex > -1) {
				map.remove(maxIndex);
				map.put(index, dist);
			}
			
		}
		return map;
	}
	
	/**
	 * calculates the index of the k nearest points
	 * @param train training data
	 * @param testRow single test row
	 * @param k number of neighbors considering
	 * @return index, distance pair map
	 */
	private static Map<Integer,Double> indexOfKNearest(Matrix train, double[] testRow, int k){
		Map<Integer,Double> indexOfKNearest = new HashMap<Integer,Double>();
		int rowLength = train.getRowDimension();
		int colLength = train.getColumnDimension();
		double[] trainRow = new double[colLength];
		double dist;
		for(int i = 0 ; i < rowLength; i++){
			trainRow = train.getMatrix(i, i, 0, colLength - 1).getRowPackedCopy();
			dist = getEuclideanDistance(trainRow,testRow);
			indexOfKNearest = replaceBiggestIndex(indexOfKNearest,i,dist,k);
		}
		return indexOfKNearest;
	}
	
	/**
	 * calculates the votes of the K nearest neighbors
	 * @param classVote class value and it's count pair
	 * @return the class value with most number of votes
	 */
	private static Integer majorityClassVote(Map<Integer,Integer> classVote){
		Integer maxVoteClass = new Integer(0);
		for(Entry<Integer, Integer> pair : classVote.entrySet()){
			if(maxVoteClass < pair.getValue()) maxVoteClass = pair.getKey();
		}
		return maxVoteClass;
	}
	
	/**
	 * the main classifier function
	 * @param train training data set
	 * @param test testing data set
	 * @param k number of neighbors to be considered
	 * @return
	 */
	public static int[] knnClassifier(Matrix train, Matrix test, int k){
		int trainRowLength = train.getRowDimension();
		int trainColLength = train.getColumnDimension();
		int testRowLength = test.getRowDimension();
		int testColLength = test.getColumnDimension();
		int[] predictedClass = new int[testRowLength];
		for(int i = 0; i < testRowLength; i++){
			double[] testRow = test.getMatrix(i, i, 0, testColLength - 2).getRowPackedCopy();
			Map<Integer,Double> indexOfKNearest = indexOfKNearest(train.getMatrix(0, trainRowLength - 1, 0, trainColLength - 2),testRow,k);
			Map<Integer,Integer> classVote = new HashMap<Integer,Integer>();
			for(Entry<Integer,Double> pair : indexOfKNearest.entrySet()){
				int index = pair.getKey();
				int classValue = (int)train.get(index, trainColLength - 1); 
				Integer count = classVote.putIfAbsent(classValue, 1);
				if(count != null){
					classVote.replace(classValue, ++count);
				}
			}
			predictedClass[i] = majorityClassVote(classVote);
		}
		return predictedClass;
	}
	
	/**
	 * randomized the input data
	 * @param data
	 * @return
	 */
	public static Matrix randomizeMatrix(Matrix data){
		Matrix randomData = data;
		int dataRowLength = randomData.getRowDimension();
		int dataColLength = randomData.getColumnDimension();
		for(int i = 0;i < dataRowLength/2;i++){
			int rand = ThreadLocalRandom.current().nextInt(0, dataRowLength/2);
			Matrix row1 = randomData.getMatrix(rand, rand, 0, dataColLength - 1);
			Matrix row2 = randomData.getMatrix(rand + dataRowLength/2, rand + dataRowLength/2, 0, dataColLength - 1);
			randomData.setMatrix(rand, rand, 0, dataColLength - 1, row2);
			randomData.setMatrix(rand + dataRowLength/2, rand + dataRowLength/2, 0, dataColLength - 1, row1);
		}
		
		return randomData;
	}
	
	/**
	 * returns the split of the training and the test data for the k-fold evaluation
	 * @param n
	 * @param k
	 * @param fold
	 * @return
	 */
	private static List<Integer[]> getTrainingTestExtractArray(int n,int k, int fold){
		int foldLB = (n/k)*(fold-1);
		int foldUB = (n/k)*(fold);
		Integer[] train = new Integer[(n/k) * (k - 1)];
		Integer[] test = new Integer[n/k];
		int trainCount = 0, testCount = 0,i = 0;
		while(testCount < test.length || trainCount < train.length){
			if(foldLB <= i && i < foldUB){
				test[testCount++] = i++;
			} else {
				train[trainCount++] = i++;
			}
		}
		List<Integer[]> split = new ArrayList<Integer[]>();
		split.add(train);
		split.add(test);
		return split;
	}
	
	/**
	 * converts the input Integer array to a primitive int array
	 * @param array
	 * @return
	 */
	public static int[] toPrimitiveArray(Integer[] array){
		int[] primArray = new int[array.length];
		for(int i = 0; i< array.length; i++){
			primArray[i] = array[i].intValue();
		}
		return primArray;
	}
	
	public static void main(String[] args) {
		try {
			Matrix data = MatrixData.getDataMatrix("data\\data.txt");
			int rowLength = data.getRowDimension();
			Matrix randomData = randomizeMatrix(data);                            //randomizing the input data
			int kFold = 5,fold = 1;
			double accuracy = 0;
			for(int i = 0; i < kFold; i++){
				List<Integer[]> split = getTrainingTestExtractArray(randomData.getRowDimension(), kFold, i + 1);
				int[] trainExtract = toPrimitiveArray(split.get(0));
				int[] testExtract = toPrimitiveArray(split.get(1));
				
				Matrix train = data.getMatrix(trainExtract, 0, data.getColumnDimension() -1);
				Matrix test = data.getMatrix(testExtract, 0, data.getColumnDimension() -1);
				int correctPrediction = 0;
				int[] predictedClass = knnClassifier(train,test,5);
				for(int j = 0;j<predictedClass.length;j++){
					if((double)predictedClass[j] == data.get(((rowLength * i)/kFold) + j, data.getColumnDimension() - 1)){
						correctPrediction++;
					}
				}
				accuracy += (double)correctPrediction/predictedClass.length;
			}
			System.out.println("Average Accuracy = " + accuracy/5);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
