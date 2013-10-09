package edu.stanford.nlp.kbp.slotfilling.classify;

import java.util.List;

import edu.stanford.nlp.math.ArrayMath;

public class Softmax {
  
  /**
   * Computes the softmax of score given the total list of scores
   * This must be computed in log space, because some of the values may be too large for exp
   * @param score
   * @param scores
   * @param gamma
   */
  public static double softmax(double score, List<Double> scores, double gamma) {
    double [] scoreArray = new double[scores.size()];
    for(int i = 0; i < scoreArray.length; i ++) 
      scoreArray[i] = gamma * scores.get(i);
    double logSoftmax = (gamma * score) - ArrayMath.logSum(scoreArray);
    double softmax = Math.exp(logSoftmax);
    return softmax;
  }
}
