package edu.stanford.nlp.kbp.slotfilling.common;

public class NormalDistribution {
  private double variance;
  private double mean;

  public NormalDistribution(double m) {
    this(1.0, m);
  }

  public NormalDistribution(double v, double m) {
    this.variance = v;
    this.mean = m;
  }
  
  public double f(double x) {
    return Math.exp(Math.pow(x - mean, 2) / (-2.0 * variance)) / Math.sqrt(2.0 * Math.PI * variance);  
  }
  
  public static void main(String[] args) {
    NormalDistribution nd = new NormalDistribution(4);
    for(double i = 0; i < 10; i += 1.0) {
      System.out.println(i + " " + nd.f(i));
    }
  }
}
