package edu.stanford.nlp.kbp.slotfilling.common;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;

public class ComputeProbabilityHistogram {
  static double [] INTERVALS = { 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0 };
  public static void main(String[] args) throws Exception {
    Counter<Double> counts = new ClassicCounter<Double>();
    BufferedReader is = new BufferedReader(new InputStreamReader(System.in));
    for(String line; (line = is.readLine()) != null; ) {
      double p = Double.valueOf(line);
      for(int i = 0; i < INTERVALS.length; i ++){
        if(p > INTERVALS[i]){
          counts.incrementCount(INTERVALS[i]);
          break;
        }
      }
    }
    List<Double> intervals = new ArrayList<Double>(counts.keySet());
    Collections.sort(intervals);
    for(Double i: intervals) {
      System.out.println(i + "\t" + (int) counts.getCount(i));
    }
  }
   
}
