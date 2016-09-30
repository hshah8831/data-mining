package edu.neu.driver;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Plsa {

    private int topicNum;

    private int docSize;

    private int vocabularySize;

    private int[][] docTermMatrix;

    //p(z|d)
    private double[][] docTopicPros;

    //p(w|z)
    private double[][] topicTermPros;

    //p(z|d,w)
    private double[][][] docTermTopicPros;

    private List<String> allWords;

    public Plsa(int numOfTopic) {
        topicNum = numOfTopic;
        docSize = 0;
    }
    
    public void setDocSize(int DocSize)
    {
    	docSize = DocSize;
    }
     
    public void readWordDict(String wordDictFile) throws NumberFormatException, IOException
    {
    	String line;
        BufferedReader br = new BufferedReader(new FileReader(wordDictFile));

        allWords = new ArrayList<String>();
        while ((line = br.readLine()) != null) {
            String rowData[] = line.split("\t");
            allWords.add(rowData[1]); //starting from 0
        }
        vocabularySize = allWords.size(); 
    }
        
    public void readDocTermMatrix(String docTermFile) throws NumberFormatException, IOException
    {
    	String line;
        BufferedReader br = new BufferedReader(new FileReader(docTermFile));
        docTermMatrix = new int[docSize][vocabularySize];
        
        while ((line = br.readLine()) != null) {
            String rowData[] = line.split("\t");
            int rowIntData[] = new int[rowData.length];

            // To convert String data into int  
            for (int i = 0; i < rowData.length; ++i) {
                rowIntData[i] = Integer.parseInt(rowData[i]);
            }
            docTermMatrix[rowIntData[0]-1][rowIntData[1]-1] = rowIntData[2];
        }
    }
    /**
     * 
     * train plsa
     * 
     * @param docs all documents
     */
    public void train(int maxIter) {

        docTopicPros = new double[docSize][topicNum];
        topicTermPros = new double[topicNum][vocabularySize];
        docTermTopicPros = new double[docSize][vocabularySize][topicNum];

        //init p(z|d),for each document the constraint is sum(p(z|d))=1.0
         for (int i = 0; i < docSize; i++) {
            double[] pros = randomProbilities(topicNum);
            for (int j = 0; j < topicNum; j++) {
                docTopicPros[i][j] = pros[j];
            }
        }
        //init p(w|z),for each topic the constraint is sum(p(w|z))=1.0
        for (int i = 0; i < topicNum; i++) {
            double[] pros = randomProbilities(vocabularySize);
            for (int j = 0; j < vocabularySize; j++) {
                topicTermPros[i][j] = pros[j];
            }
        }

        //use em to estimate params
        for (int i = 0; i < maxIter; i++) {
            em();
//            System.out.print(i+"\n");
        }
        System.out.println("done");
    }

    /**
     * 
     * EM algorithm
     * 
     */
    private void em() {
        /*
         * E-step,calculate posterior probability p(z|d,w,&),& is
         * model params(p(z|d),p(w|z))
         * 
         * p(z|d,w,&)=p(z|d)*p(w|z)/sum(p(z'|d)*p(w|z'))
         * z' represent all possible topic
         * 
         */
        for (int docIndex = 0; docIndex < docSize; docIndex++) {
            for (int wordIndex = 0; wordIndex < vocabularySize; wordIndex++) {
                double total = 0.0;
                double[] perTopicPro = new double[topicNum];
                for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {	
                    double numerator = docTopicPros[docIndex][topicIndex] * topicTermPros[topicIndex][wordIndex]; 
                    total += numerator;
                    perTopicPro[topicIndex] = numerator;
                }

                if (total == 0.0) {
                    total = avoidZero(total);
                }

                for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
                    docTermTopicPros[docIndex][wordIndex][topicIndex] = perTopicPro[topicIndex] / total;
                }
            }
        }

        //M-step
        /*
         * update p(w|z),p(w|z)=sum(n(d',w)*p(z|d',w,&))/sum(sum(n(d',w')*p(z|d',w',&)))
         * 
         * d' represent all documents
         * w' represent all vocabularies
         * 
         * 
         */
        for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
            double totalDenominator = 0.0;
            for (int wordIndex = 0; wordIndex < vocabularySize; wordIndex++) {
                double numerator = 0.0;
                for (int docIndex = 0; docIndex < docSize; docIndex++) {
                    numerator += docTermMatrix[docIndex][wordIndex]
                            * docTermTopicPros[docIndex][wordIndex][topicIndex];
                }

                topicTermPros[topicIndex][wordIndex] = numerator;

                totalDenominator += numerator;
            }

            if (totalDenominator == 0.0) {
                totalDenominator = avoidZero(totalDenominator);
            }

            for (int wordIndex = 0; wordIndex < vocabularySize; wordIndex++) {
                topicTermPros[topicIndex][wordIndex] = topicTermPros[topicIndex][wordIndex]
                        / totalDenominator;
            }
        }
        /*
         * update p(z|d),p(z|d)=sum(n(d,w')*p(z|d,w'&))/sum(sum(n(d,w')*p(z'|d,w',&)))
         * 
         * w' represent all vocabularies
         * z' represent all topics
         * 
         */
        for (int docIndex = 0; docIndex < docSize; docIndex++) {
            //actually equal sum(w) of this doc
            double totalDenominator = 0.0;
            for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
                double numerator = 0.0;
                for (int wordIndex = 0; wordIndex < vocabularySize; wordIndex++) {
                    numerator += docTermMatrix[docIndex][wordIndex]
                            * docTermTopicPros[docIndex][wordIndex][topicIndex];
                }
                docTopicPros[docIndex][topicIndex] = numerator;
                totalDenominator += numerator;
            }

            if (totalDenominator == 0.0) {
                totalDenominator = avoidZero(totalDenominator);
            }

            for (int topicIndex = 0; topicIndex < topicNum; topicIndex++) {
                docTopicPros[docIndex][topicIndex] = docTopicPros[docIndex][topicIndex]
                        / totalDenominator;
            }
        }
    }

//    private List<String> statisticsVocabularies(List<Document> docs) {
//        Set<String> uniqWords = new HashSet<String>();
//        for (Document doc : docs) {
//            for (String word : doc.getWords()) {
//                if (!uniqWords.contains(word)) {
//                    uniqWords.add(word);
//                }
//            }
//            docSize++;
//        }
//        vocabularySize = uniqWords.size();
//
//        return new LinkedList<String>(uniqWords);
//    }

    /**
     * 
     * 
     * Get a normalize array
     * 
     * @param size
     * @return
     */
    public double[] randomProbilities(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("The size param must be greate than zero");
        }
        double[] pros = new double[size];

        int total = 0;
        Random r = new Random();
        for (int i = 0; i < pros.length; i++) {
            //avoid zero
            pros[i] = r.nextInt(size) + 1;

            total += pros[i];
        }

        //normalize
        for (int i = 0; i < pros.length; i++) {
            pros[i] = pros[i] / total;
        }

        return pros;
    }

    /**
     * 
     * @return
     */
    public double[][] getDocTopics() {
        return docTopicPros;
    }

    /**
     * 
     * @return
     */
    public double[][] getTopicWordPros() {
        return topicTermPros;
    }

    /**
     * 
     * @return
     */
    public List<String> getAllWords() {
        return allWords;
    }

    /**
     * 
     * Get topic number
     * 
     * 
     * @return
     */
    public Integer getTopicNum() {
        return topicNum;
    }

    /**
     * 
     * Get p(w|z)
     * 
     * @param word
     * @return
     */
    public double[] getTopicWordPros(String word) {
        int index = allWords.indexOf(word);
        if (index != -1) {
            double[] topicWordPros = new double[topicNum];
            for (int i = 0; i < topicNum; i++) {
                topicWordPros[i] = topicTermPros[i][index];
            }
            return topicWordPros;
        }

        return null;
    }

    /**
     * 
     * avoid zero number.if input number is zero, we will return a magic
     * number.
     * 
     * 
     */
    private final static double MAGICNUM = 0.0000000000000001;

    public double avoidZero(double num) {
        if (num == 0.0) {
            return MAGICNUM;
        }

        return num;
    }
}