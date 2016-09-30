package edu.neu.driver;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class Main {

	public static void main(String[] args) throws NumberFormatException, IOException {
		// TODO Auto-generated method stub
		int numoftopics = 4;
		int DocSize = 20;
		String wordDictFile = "Data\\term_dict.txt";
		String docTermFile = "Data\\CT.txt";
		Plsa myPLSA = new Plsa(numoftopics);
		myPLSA.setDocSize(DocSize);
		myPLSA.readWordDict(wordDictFile);
		myPLSA.readDocTermMatrix(docTermFile);
		int maxIter = 1000;
		myPLSA.train(maxIter);	
		double[][] theta = myPLSA.getDocTopics();
		double[][] beta = myPLSA.getTopicWordPros();
		System.out.println("Theta dimentions " + theta.length + " * " + theta[0].length);
		int[] groundTruth = {3,3,1,1,1,4,3,3,4,2,4,2,1,2,3,2,1,2,4,4};
		int[] cluster = new int[20];
		
		//assigning topic to documents
		for(int i = 0; i < theta.length; i++){
			double max = theta[i][0];
			int maxIndex = 0;
			for(int j = 0; j < theta[i].length ; j++){
				if(theta[i][j] > max){
					max = theta[i][j];
					maxIndex = j;
				}
			}
			cluster[i] = maxIndex;
			System.out.println("Document " + i + "is from topic " + maxIndex);		
		}

		//finding top ten words in each topic
		for(int i = 0; i < beta.length; i++){
			System.out.println("TOPIC " + i);
			if(beta[i].length > 10){
				Map<Integer, Double> max = new HashMap<Integer, Double>();
				for(int j = 0; j < 10; j++){
					max.put(j, beta[i][j]);
				}
				for(int j = 11; j < beta[i].length; j++){
					int indexLowest = lowestOf(max);
					if(beta[i][j] < max.get(indexLowest)){
						max.remove(indexLowest);
						max.put(j, beta[i][j]);
					}
				}
				for(Map.Entry<Integer, Double> pair : max.entrySet()){
					System.out.print(" " + myPLSA.getAllWords().get(pair.getKey()));
				}
			} else {
				for(int j = 0; j < beta[i].length; j++){
					System.out.print(" " + myPLSA.getAllWords().get(j));
				}
			}
			System.out.println(" ");
		}
		
		//calculating the NMI
		int[][] purityMat = new int[4][4];
		int[] groundTruthArr = new int[4];
		int[] clusterArr = new int[4];
		for(int i = 0; i < 20; i++){
			purityMat[cluster[i]][groundTruth[i] - 1]++;
			groundTruthArr[groundTruth[i] - 1]++;
			clusterArr[cluster[i]]++;
		}
		
		int N = 20;
		double information = 0.0d;
		double entropyOmega = 0.0d;
		double entropyCluster = 0.0d;
		double NMI = 0.0d;
		for(int i = 0; i < 4; i++){
			for(int j = 0; j < 4; j++){
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
		System.out.println(" NMI = " + NMI);
		
	}
	
	public static int lowestOf(Map<Integer, Double> max){
		int ret = 0;
		double min = 99999999.99;
		for(Map.Entry<Integer, Double> pair : max.entrySet()){
			if(pair.getValue() < min){
				min = pair.getValue();
				ret = pair.getKey();
			}
		}
		return ret;
	}

}
