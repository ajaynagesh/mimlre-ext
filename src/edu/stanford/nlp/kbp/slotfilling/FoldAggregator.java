package edu.stanford.nlp.kbp.slotfilling;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * Computes average results given 10 slot-filling folds
 * @author Mihai
 *
 */
public class FoldAggregator {
  static class Score {
    double p;
    double pStdev;
    double r;
    double rStdev;
    double f1;
    double f1Stdev;
    
    public Score() {
      this.p = 0;
      this.r = 0;
      this.f1 = 0;
      this.pStdev = 0;
      this.rStdev = 0;
      this.f1Stdev = 0;
    }
    
    public Score(double p, double r, double f1) {
      this.p = p;
      this.r = r;
      this.f1 = f1;
      this.pStdev = 0;
      this.rStdev = 0;
      this.f1Stdev = 0;
    }
    
    public String toString() {
      if(pStdev == 0 && rStdev == 0 && f1Stdev == 0){
        return "P\t" + pretty(p) + "\tR\t" + pretty(r) + "\tF1\t" + pretty(f1);
      } else {
        return pretty(p) + "\t" + pretty(pStdev) + 
          "\t" + pretty(r) + "\t" + pretty(rStdev) + 
          "\t" + pretty(f1) + "\t" + pretty(f1Stdev);
      }
    }
    
    static double pretty(double v) {
      double d = v * 100;
      DecimalFormat twoDForm = new DecimalFormat("#.##");
      return Double.valueOf(twoDForm.format(d));
    }
  }
  
  public static void main(String[] args) throws Exception {
    if(args.length != 2) {
      System.err.println("Usage: java edu.stanford.nlp.kbp.slotfilling.FoldAggregator <TOP DIR> <ID>");
      System.exit(1);
    }
    String topDir = args[0];
    String id = args[1];
    
    List<Score> scores = new ArrayList<Score>();
    for(int fold = 0; fold < 10; fold ++) {
      Score s = fetchScore(topDir + File.separator + Integer.toString(fold), id);
      scores.add(s);
      System.out.println("Fold " + fold + ": " + s);
    }
    
    Score avg = average(scores);
    System.out.println(id + "\t" + avg);
  }
  
  static Score average(List<Score> scores) {
    Score avg = new Score();
    for(Score s: scores) {
      avg.p += s.p;
      avg.r += s.r;
      avg.f1 += s.f1;
    }
    avg.p /= (double) scores.size();
    avg.r /= (double) scores.size();
    avg.f1 /= (double) scores.size();
    
    for(Score s: scores) {
      avg.pStdev += (avg.p - s.p) * (avg.p - s.p);
      avg.rStdev += (avg.r - s.r) * (avg.r - s.r);
      avg.f1Stdev += (avg.f1 - s.f1) * (avg.f1 - s.f1);
    }
    avg.pStdev /= (double) scores.size();
    avg.rStdev /= (double) scores.size();
    avg.f1Stdev /= (double) scores.size();
    avg.pStdev = Math.sqrt(avg.pStdev);
    avg.rStdev = Math.sqrt(avg.rStdev);
    avg.f1Stdev = Math.sqrt(avg.f1Stdev);
    return avg;
  }
  
  static Score fetchScore(String dir, String id) throws Exception {
    BufferedReader is = new BufferedReader(new FileReader(dir + File.separator + id + ".query_score_t0.txt"));
    double p = -1, r = -1, f1 = -1;
    for(String line; (line = is.readLine()) != null; ) {
      String [] bits = line.trim().split("\\s+");
      if(bits[0].equals("Precision:")){
        p = Double.valueOf(bits[bits.length - 1]);
      } else if(bits[0].equals("Recall:")) {
        r = Double.valueOf(bits[bits.length - 1]);
      } else if(bits[0].equals("F1:")) {
        f1 = Double.valueOf(bits[bits.length - 1]);
      }
    }
    is.close();
    assert(p != -1);
    assert(r != -1);
    assert(f1 != -1);
    return new Score(p, r, f1);
  }
}
