package edu.stanford.nlp.kbp.slotfilling.classify;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.Pair;

/**
 * Multi-class local LR classifier with incomplete negatives
 * This is what we used at KBP 2011
 */
public class OneVsAllRelationExtractor extends RelationExtractor {
  private static final long serialVersionUID = -9156368776695470568L;
    
  private Map<String, LinearClassifier<String, String>> classifiers = null;
  
  /** Regularization coeficient */
  private double sigma;
  
  /** Softmax parameter */
  private double gamma;
  
  public void setSigma(double d) { sigma = d; }
  public void setGamma(double d) { gamma = d; }
  
  public OneVsAllRelationExtractor() {
    this(1.0, Constants.SOFTMAX_GAMMA);
  }
  
  public OneVsAllRelationExtractor(double sigma, double gamma) {
    super();
    this.sigma = sigma;
    this.gamma = gamma;
  }
  
  @Override
  public Counter<String> classifyMentions(List<Collection<String>> relation) {
    assert(classifiers != null);

    Counter<String> labels = new ClassicCounter<String>();
    for(Collection<String> mention: relation) {
      // System.err.println("Classifying slot " + mention.mention().getArg(1).getExtentString());
      Datum<String, String> datum = new BasicDatum<String, String>(mention);
      Pair<String, Double> label = annotateDatum(datum);
      if(! label.first().equals(RelationMention.UNRELATED)) {
        // System.err.println("Classified slot " + mention.mention().getArg(1).getExtentString() + " with label " + label.first() + " with score " + label.second());
        labels.incrementCount(label.first(), label.second());
      }
    }
    
    return labels;
  }

  private Pair<String, Double> annotateDatum(Datum<String, String> testDatum) {
    Set<String> knownLabels = classifiers.keySet();
    
    // fetch all scores 
    List<Pair<String, Double>> allLabelScores = new ArrayList<Pair<String,Double>>();
    List<Double> scores = new ArrayList<Double>();
    for(String knownLabel: knownLabels){
      LinearClassifier<String, String> labelClassifier = classifiers.get(knownLabel);
      Pair<String, Double> pred = classOf(testDatum, labelClassifier);  
      assert(pred != null);
      if(pred.second > 0.5) allLabelScores.add(pred);
      scores.add(pred.second);
    }
    
    // convert scores to probabilities using softmax
    for(Pair<String, Double> ls: allLabelScores){
      ls.second = Softmax.softmax(ls.second, scores, gamma);
    }
    
    Collections.sort(allLabelScores, new Comparator<Pair<String, Double>>() {
      @Override
      public int compare(Pair<String, Double> o1, Pair<String, Double> o2) {
        if(o1.second > o2.second) return -1;
        if(o1.second == o2.second) return 0;
        return 1;
      }
    });
    
    if(allLabelScores.size() > 0) return allLabelScores.iterator().next();
    return new Pair<String, Double>(RelationMention.UNRELATED, 1.0);
  }
  
  private Pair<String, Double> classOf(Datum<String, String> datum, LinearClassifier<String, String> classifier) {
    Counter<String> probs = classifier.probabilityOf(datum); 
    List<Pair<String, Double>> sortedProbs = Counters.toDescendingMagnitudeSortedListWithCounts(probs);
    for(Pair<String, Double> ls: sortedProbs){
      if(! ls.first.equals(RelationMention.UNRELATED)) return ls;
    }
    return null;
  }

  public void save(String modelpath) throws IOException {
    // make sure the modelpath directory exists
    int lastSlash = modelpath.lastIndexOf(File.separator);
    if(lastSlash > 0){
      String path = modelpath.substring(0, lastSlash);
      File f = new File(path);
      if (! f.exists()) {
        f.mkdirs();
      }
    }
    
    FileOutputStream fos = new FileOutputStream(modelpath);
    ObjectOutputStream out = new ObjectOutputStream(fos);
    
    assert(classifiers != null);
    out.writeObject(classifiers);
    out.writeDouble(sigma);
    
    out.close(); 
  }
  
  public static RelationExtractor load(String modelPath) throws IOException, ClassNotFoundException {
    InputStream is = new FileInputStream(modelPath);
    ObjectInputStream in = new ObjectInputStream(is);
    OneVsAllRelationExtractor ex = null;

    Map<String, LinearClassifier<String, String>> classifiers = 
      ErasureUtils.<Map<String, LinearClassifier<String, String>>>uncheckedCast(in.readObject());
    double sigma = in.readDouble();

    ex = new OneVsAllRelationExtractor();
    ex.classifiers = classifiers;
    ex.sigma = sigma;
    
    in.close();
    is.close();
    return ex;
  }

  public void train(Map<String, GeneralDataset<String, String>> trainSets){
    Set<String> labels = trainSets.keySet();
    logger.severe("WILL TRAIN " + labels.size() + " MODELS.");
    
    classifiers = new HashMap<String, LinearClassifier<String,String>>();
    for(String label: labels) {
      logger.severe("TRAINING CLASSIFIER FOR LABEL: " + label);
      LinearClassifier<String,String> labelClassifier = trainOne(trainSets.get(label));
      classifiers.put(label, labelClassifier);
    }
    logger.severe("Finished training all classifiers.");
  }

  private LinearClassifier<String, String> trainOne(GeneralDataset<String, String> trainSet) {
    LinearClassifierFactory<String, String> lcFactory = 
      new LinearClassifierFactory<String, String>(1e-4, false, sigma); // new QNMinimizer(15), 1e-4, false, LogPrior.LogPriorType.HUBER.ordinal(), sigma);
    // lcFactory.useHybridMinimizer();
    // lcFactory.useInPlaceStochasticGradientDescent();
    lcFactory.setVerbose(false);
    LinearClassifier<String, String> localClassifier = lcFactory.trainClassifier(trainSet);
    return localClassifier;
  }
}
