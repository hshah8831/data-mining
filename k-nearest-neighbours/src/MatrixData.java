package org.neu.main;

import Jama.Matrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads the data form CSV files and converts it into Matrix data
 */
public class MatrixData {
    public static Matrix getDataMatrix(String filePath) throws IOException {
        List<double[]> listData = new ArrayList<>();
        String line;
        boolean flag = false;

        BufferedReader br = new BufferedReader(new FileReader(filePath));
        while ((line = br.readLine()) != null) {
            // To skip the first row in the files
            if(flag == true) {
                flag=false;
                continue;
            }
            String rowData[] = line.split("\t");
            double rowDoubleData[] = new double[rowData.length];

            // To convert String data into double
            for (int i = 0; i < rowData.length; ++i) {
                rowDoubleData[i] = Double.parseDouble(rowData[i]);
            }
            listData.add(rowDoubleData);
        }
        int data_cols = listData.get(0).length;
        int data_rows = listData.size();

        Matrix matrixData = new Matrix(data_rows, data_cols);

        // storing data in Matrix
        for (int r = 0; r < data_rows; r++) {
            for (int c = 0; c < data_cols ; c++) {
                matrixData.set(r, c, listData.get(r)[c]);
            }
        }
        return matrixData;
    }
}
