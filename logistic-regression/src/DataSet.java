package edu.neu.main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class DataSet {

 public static List < Instance > readDataSet(String file) throws FileNotFoundException {
  List < Instance > dataset = new ArrayList < Instance > ();
  Scanner scanner = new Scanner(new File(file));
  while (scanner.hasNextLine()) {
   String line = scanner.nextLine();
   if (line.startsWith("@") || line.isEmpty()) {
    continue;
   }
   String[] columns = line.split("	");

   double[] data = new double[columns.length];
   int i = 0;
   data[0] = 1;
   for (i = 0; i < columns.length - 1; i++) {
    data[i + 1] = Double.parseDouble(columns[i]);
   }
   int label = Integer.parseInt(columns[i]);
   Instance instance = new Instance(label, data);
   dataset.add(instance);
  }
  return dataset;
 }
}