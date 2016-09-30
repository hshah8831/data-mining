package edu.neu.main;

public class Instance {
 public int label;
 public double[] x;
 public int dimension;

 public Instance(int label, double[] x) {
  this.label = label;
  this.x = x;
  dimension = x.length;
 }

 public int getLabel() {
  return label;
 }

 public double[] getX() {
  return x;
 }

 public int getDimension() {
  return dimension;
 }
}