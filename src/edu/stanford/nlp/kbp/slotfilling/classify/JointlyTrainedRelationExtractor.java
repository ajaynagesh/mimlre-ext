package edu.stanford.nlp.kbp.slotfilling.classify;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Triple;

public abstract class JointlyTrainedRelationExtractor extends RelationExtractor {
  private static final long serialVersionUID = 1L;

  public abstract void train(MultiLabelDataset<String, String> datums);
  
  public static Triple<Double, Double, Double> score(
      List<Set<String>> goldLabels,
      List<Counter<String>> predictedLabels) {
    int total = 0, predicted = 0, correct = 0;
    for(int i = 0; i < goldLabels.size(); i ++) {
      Set<String> gold = goldLabels.get(i);
      Counter<String> preds = predictedLabels.get(i);
      total += gold.size();
      predicted += preds.size();
      for(String label: preds.keySet()) {
        if(gold.contains(label)) correct ++;
      }
    }
    
    double p = (double) correct / (double) predicted;
    double r = (double) correct / (double) total;
    double f1 = (p != 0 && r != 0 ? 2*p*r/(p+r) : 0);
    return new Triple<Double, Double, Double>(p, r, f1);
  }
  
  public Triple<Double, Double, Double> test(
      List<List<Collection<String>>> relations,
      List<Set<String>> goldLabels,
      List<Counter<String>> predictedLabels) {
    if(predictedLabels == null)
      predictedLabels = new ArrayList<Counter<String>>();
    for(int i = 0; i < relations.size(); i ++) {
      List<Collection<String>> rel = relations.get(i);
      Counter<String> preds = classifyMentions(rel);
      predictedLabels.add(preds);
    }
    return score(goldLabels, predictedLabels);
  }
  
  public Triple<Double, Double, Double> oracle(
      List<List<Collection<String>>> relations,
      List<Set<String>> goldLabels,
      List<Counter<String>> predictedLabels) {
    if(predictedLabels == null)
      predictedLabels = new ArrayList<Counter<String>>();
    for(int i = 0; i < relations.size(); i ++) {
      List<Collection<String>> rel = relations.get(i);
      Counter<String> preds = classifyOracleMentions(rel, goldLabels.get(i));
      predictedLabels.add(preds);
    }
    return score(goldLabels, predictedLabels);
  }
  
  public abstract void save(String path) throws IOException;
  public abstract void load(ObjectInputStream in) throws IOException, ClassNotFoundException;
}
