package edu.stanford.nlp.kbp.slotfilling.classify;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.Triple;

/**
 * Other variants of a latent-variable averaged perceptron classifier for relation classification
 * This includes:
 * - Boring local perceptron
 * - Joint and local models with incomplete negatives (in the spirit of our KBP 2011 paper)
 *
 * @author Mihai
 *
 */
public class PerceptronExtractor extends JointlyTrainedRelationExtractor {
  private static final long serialVersionUID = 1L;
  
  private static boolean SOFT_UNKNOWN = false;
  
  /** 
   * Stores weight information for one label
   * @author Mihai
   *
   */
  static class LabelWeights implements Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * Weights for a binary classifier vector (for one label)
     * This stores the expanded vector for all known features
     */
    double [] weights;
    
    /** Indicates how many iterations has this vector survived */
    int survivalIterations;
    
    /** 
     * Average vector computed as a weighted sum of all seen vectors
     * The weight for each vector is the number of iterations it survived 
     */
    double [] avgWeights;
    
    LabelWeights(int numFeatures) {
      weights = new double[numFeatures];
      Arrays.fill(weights, 0.0);
      survivalIterations = 0;
      avgWeights = new double[numFeatures];
      Arrays.fill(avgWeights, 0.0);
    }
    
    void clear() {
      weights = null;
    }
    
    void updateSurvivalIterations() {
      survivalIterations ++;
    }
    
    /** Adds the latest weight vector to the average vector */
    private void addToAverage() {
      double confidenceInThisVector = survivalIterations;
      for(int i = 0; i < weights.length; i ++){
        avgWeights[i] += weights[i] * confidenceInThisVector;
      }
    }
    
    void update(int [] datum, double weight) {
      // add this vector to the avg
      addToAverage();
      
      // actual update
      for(int d: datum){
        if(d > weights.length) expand();
        weights[d] += weight;
      }
      
      // this is a new vector, so let's reset its survival counter
      survivalIterations = 0;
    }
    
    private void expand() {
      throw new RuntimeException("ERROR: LabelWeights.expand() not supported yet!");
    }
    
    double dotProduct(Counter<Integer> vector) {
      return dotProduct(vector, weights);
    }
    
    void normalize(double norm) {
      if(norm > 0){
        for(int i = 0; i < avgWeights.length; i ++) 
          avgWeights[i] /= norm;
      }
    }
    
    double avgDotProduct(Collection<String> features, Index<String> featureIndex) {
      Counter<Integer> vector = new ClassicCounter<Integer>();
      for(String feat: features) {
        int idx = featureIndex.indexOf(feat);
        if(idx >= 0) vector.incrementCount(idx);
      }
      
      return dotProduct(vector, avgWeights);
    }
    
    static double dotProduct(Counter<Integer> vector, double [] weights) {
      double dotProd = 0;
      for (Map.Entry<Integer, Double> entry : vector.entrySet()) {
        if(entry.getKey() == null) throw new RuntimeException("NULL key in " + entry.getKey() + "/" + entry.getValue());
        if(entry.getValue() == null) throw new RuntimeException("NULL value in " + entry.getKey() + "/" + entry.getValue());
        if(weights == null) throw new RuntimeException("NULL weights!");
        if(entry.getKey() < 0 || entry.getKey() >= weights.length) throw new RuntimeException("Invalid key " + entry.getKey() + ". Should be >= 0 and < " + weights.length);
        dotProd += entry.getValue() * weights[entry.getKey()];
      }
      return dotProd;
    }
  }
  
  /** Stores weight information for each known Z label (including NIL) */
  LabelWeights [] zWeights;
  
  Index<String> labelIndex;
  Index<String> zFeatureIndex;
  /** Index of the NIL label */
  int nilIndex;
  
  /** Joint or local model? */
  final ModelType modelType;
  /** Number of epochs during training */
  final int epochs;
  /** If true, convert scores to probabilities using softmax */
  final boolean softmaxEnabled;
  /** Softmax parameter to generate probabilities from scores */
  final double gamma;
  /** Enable very verbose debug output */
  final boolean verbose;
  
  private Counter<Integer> posUpdateStats;
  private Counter<Integer> negUpdateStats;
  private Counter<Integer> unknownUpdateStats;
  
  public PerceptronExtractor(Properties props) throws IOException {
    Log.severe("PerceptronExtractor configured with the following properties:");
    this.epochs = PropertiesUtils.getInt(props, Props.PERCEPTRON_EPOCHS, 10);
    Log.severe("epochs = " + epochs);
    this.softmaxEnabled = PropertiesUtils.getBool(props, Props.PERCEPTRON_SOFTMAX, true);
    Log.severe("softmaxEnabled = " + softmaxEnabled);
    String normType = props.getProperty(Props.PERCEPTRON_NORMALIZE, "L2J");
    Log.severe("normType = " + normType);
    String modType = props.getProperty(Props.MODEL_TYPE, Constants.DEFAULT_ATLEASTONCE_MODEL);
    this.modelType = ModelType.stringToModel(modType);
    Log.severe("modelType = " + modelType);
    this.gamma = Constants.SOFTMAX_GAMMA;
    Log.severe("gamma = " + gamma);
    this.verbose = false;
  }
  
  public void save(String modelPath) throws IOException {
    // make sure the modelpath directory exists
    int lastSlash = modelPath.lastIndexOf(File.separator);
    if(lastSlash > 0){
      String path = modelPath.substring(0, lastSlash);
      File f = new File(path);
      if (! f.exists()) {
        f.mkdirs();
      }
    }
    
    for(LabelWeights zw: zWeights) {
      zw.clear();
    }
    
    FileOutputStream fos = new FileOutputStream(modelPath);
    ObjectOutputStream out = new ObjectOutputStream(fos);
    
    assert(zWeights != null);
    out.writeInt(zWeights.length);
    for(LabelWeights zw: zWeights) {
      out.writeObject(zw);
    }
    
    out.writeObject(labelIndex);
    out.writeObject(zFeatureIndex);
    
    out.close(); 
  }
  
  public void load(ObjectInputStream in) throws IOException, ClassNotFoundException {
    int length = in.readInt();
    zWeights = new LabelWeights[length];
    for(int i = 0; i < zWeights.length; i ++){
      zWeights[i] = ErasureUtils.uncheckedCast(in.readObject());
    }
    
    labelIndex = ErasureUtils.uncheckedCast(in.readObject());
    nilIndex = labelIndex.indexOf(RelationMention.UNRELATED);
    zFeatureIndex = ErasureUtils.uncheckedCast(in.readObject());
  }
  
  public static RelationExtractor load(String modelPath, Properties props) throws IOException, ClassNotFoundException {
    InputStream is = new FileInputStream(modelPath);
    ObjectInputStream in = new ObjectInputStream(is);
    PerceptronExtractor ex = new PerceptronExtractor(props);
    ex.load(in);
    in.close();
    is.close();
    return ex;
  }
  
  @Override
  public void train(MultiLabelDataset<String, String> dataset) {
    Log.severe("Training the \"at least once\" model using "
        + dataset.featureIndex().size() + " features and "
        + "the following labels: " + dataset.labelIndex().toString());
    
    labelIndex = dataset.labelIndex();
    // add the NIL label
    labelIndex.add(RelationMention.UNRELATED);
    nilIndex = labelIndex.indexOf(RelationMention.UNRELATED);
    zFeatureIndex = dataset.featureIndex();
    
    zWeights = new LabelWeights[labelIndex.size()];
    for(int i = 0; i < zWeights.length; i ++) 
      zWeights[i] = new LabelWeights(dataset.featureIndex().size());
    
    int iterations = 0;
    for(int t = 0; t < epochs; t ++){
      // randomize the data set in each epoch
      // use a fixed seed for replicability
      Log.severe("Started epoch #" + t + "...");
      dataset.randomize(t);
      
      posUpdateStats = new ClassicCounter<Integer>();
      negUpdateStats = new ClassicCounter<Integer>();
      unknownUpdateStats = new ClassicCounter<Integer>();
      
      for(int i = 0; i < dataset.size(); i ++){
        int [][] crtGroup = dataset.getDataArray()[i];
        Set<Integer> goldPos = dataset.getPositiveLabelsArray()[i];
        Set<Integer> goldNeg = dataset.getNegativeLabelsArray()[i];
        iterations ++;
        if(verbose) inputStats(i, goldPos, goldNeg, crtGroup);
        
        if(modelType == ModelType.AT_LEAST_ONCE_INC) {
          trainJointlyOneGroupIncomplete(crtGroup, goldPos, goldNeg);
        } else if(modelType == ModelType.PERCEPTRON) {
          trainLocallyOneGroup(crtGroup, goldPos, goldNeg, false);
        } else if(modelType == ModelType.PERCEPTRON_INC) {
          trainLocallyOneGroup(crtGroup, goldPos, goldNeg, true);
        } else {
          throw new RuntimeException("Unsupported model type: " + modelType);
        }
        
        if(verbose){
          System.err.println("Group #" + i + " completed.");
          System.err.println("=============================================================");
        }
        
        for(LabelWeights zw: zWeights) {
          zw.updateSurvivalIterations();
        }
      }
      
      Log.severe("Epoch #" + t + " completed. Inspected " + 
          dataset.size() + " datum groups. Performed " +
          posUpdateStats.getCount(LABEL_ALL) + " ++ updates and " +
          negUpdateStats.getCount(LABEL_ALL) + " -- updates and " +
          unknownUpdateStats.getCount(LABEL_ALL) + " unknown updates.");
            
      // compute performance over the training set
      computeTrainingPerformance(dataset);
    } 
    Log.severe("Run model through " + iterations + " iterations.");
    
    // normalize the avg vector by the total number of iterations
    // otherwise, the average weights are too large
    for(LabelWeights zw: zWeights) {
      zw.normalize((double) iterations);
    }
    printAvgVectors();
  }
  
  private void trainLocallyOneGroup(int [][] crtGroup,
      Set<Integer> goldPos, 
      Set<Integer> goldNeg,
      boolean incompleteModel) {
    // zs - all labels with non-zero scores for each datum
    List<Counter<Integer>> zs = estimateZ(crtGroup);
    
    for(int i = 0; i < crtGroup.length; i ++){
      int [] datum = crtGroup[i];
      List<Pair<Integer, Double>> predictions = Counters.toDescendingMagnitudeSortedListWithCounts(zs.get(i));
      int prediction = nilIndex;
      if(predictions.size() > 0){
        prediction = predictions.get(0).first();
      }
      
      // positive update(s)
      for(int gold: goldPos) {
        if(gold != nilIndex && gold != prediction) {
          zWeights[gold].update(datum, +1.0);
          if(verbose) System.err.println("Update +++ on label " + gold);
          posUpdateStats.incrementCount(gold);
          posUpdateStats.incrementCount(LABEL_ALL);
        }
      }
      
      // positive update for NIL
      if(goldPos.size() == 0 && prediction != nilIndex){
        zWeights[nilIndex].update(datum, +1.0);
        if(verbose) System.err.println("Update +++ on label NIL");
        posUpdateStats.incrementCount(nilIndex);
      }
      
      // negative update
      if(prediction != nilIndex && ! goldPos.contains(prediction) && 
          (! incompleteModel || goldNeg.contains(prediction))){
        zWeights[prediction].update(datum, -1.0);
        if(verbose) System.err.println("Update --- on label " + prediction);
        negUpdateStats.incrementCount(prediction);
        negUpdateStats.incrementCount(LABEL_ALL);
      }
      
      // negative update for NIL
      if(prediction == nilIndex && goldPos.size() != 0){
        zWeights[nilIndex].update(datum, -1.0);
        if(verbose) System.err.println("Update --- on label NIL");
        negUpdateStats.incrementCount(prediction);
      }
    }
  }
  
  private int [] generateZUnknown(
      int [] zs,
      Set<Integer> goldPos, 
      Set<Integer> goldNeg) {
    int [] zUnknown = Arrays.copyOf(zs, zs.length);
    zUnknown = removeGold(zUnknown, goldNeg);
    zUnknown = removeGold(zUnknown, goldPos);
    if(verbose) System.err.println("zUnknown after removing y+/-: " + arrayToString(zUnknown));
    return zUnknown;
  }
  
  private void trainJointlyOneGroupIncomplete(
      int [][] crtGroup,
      Set<Integer> goldPos, 
      Set<Integer> goldNeg) {
    // zs - all labels with non-zero scores for each datum
    List<Counter<Integer>> zs = estimateZ(crtGroup);
    int [] zPredicted = generateZPredicted(zs);
    if(verbose) predictionZStats(zPredicted);

    // yPredicted - Y labels predicted using the current Zs
    Counter<Integer> yPredicted = estimateY(zPredicted);
    if(verbose) predictionYStats(yPredicted.keySet());

    if(updateCondition(yPredicted.keySet(), goldPos, goldNeg)){
      Set<Integer> yUpdate = computeUpdateY(yPredicted.keySet(), goldPos, goldNeg);
      if(verbose) System.err.println("yUpdate: " + yUpdate);

      int [] zUnknown = generateZUnknown(zPredicted, goldPos, goldNeg);
      Set<Integer> [] zUpdate = generateZUpdate(zUnknown, goldPos, zs);
      if(verbose) System.err.println("zUpdate after adding y+: " + arrayToString(zUpdate));
      
      Set<Integer> unknownY = null;
      List<Counter<Integer>> zProbabilities = null;
      if(SOFT_UNKNOWN) {
        // build the list of unknown Y labels for this group
        unknownY = new HashSet<Integer>();
        for(Set<Integer> zu: zUpdate) {
          for(Integer l: zu) {
            if(! goldPos.contains(l) && ! goldNeg.contains(l)) {
              unknownY.add(l);
            }
          }
        }
        if(verbose) System.err.println("yUnknown: " + unknownY);
        
        // compute the prediction probabilities for each mention in this group using softmax
        zProbabilities = new ArrayList<Counter<Integer>>();
        for(Counter<Integer> z: zs) {
          zProbabilities.add(toProbabilities(z));
        }
      }

      updateZModel(zUpdate, zPredicted, crtGroup, unknownY, zProbabilities);      
    } else {
      if(verbose) System.err.println("No update necessary.");
    }
  }
  
  /**
   * Converts a set of scores to probabilities using softmax
   */
  private Counter<Integer> toProbabilities(Counter<Integer> scores) {
    List<Double> allScores = new ArrayList<Double>();
    for(Integer l: scores.keySet()) {
      allScores.add(scores.getCount(l));
    }
    Counter<Integer> probs = new ClassicCounter<Integer>();
    for(Integer l: scores.keySet()) {
      double score = scores.getCount(l);
      double prob = Softmax.softmax(score, allScores, gamma);
      probs.setCount(l, prob);
    }
    return probs;
  }
  
  private void printAvgVectors() {
    for(int i = 0; i < zWeights.length; i ++){
      System.err.print("AVG VECTOR #" + i + ":");
      for(int j = 0; j < zWeights[i].avgWeights.length; j ++){
        double v = zWeights[i].avgWeights[j];
        if(v != 0){
          System.err.print(" " + j + ":" + v);
        }
      }
      System.err.println();
    }
  }
  
  private void computeTrainingPerformance(MultiLabelDataset<String, String> dataset) {
    Counter<Integer> total = new ClassicCounter<Integer>();
    Counter<Integer> predicted = new ClassicCounter<Integer>();
    Counter<Integer> correct = new ClassicCounter<Integer>();
    Set<Integer> seenLabels = new HashSet<Integer>();
    
    for(int i = 0; i < dataset.size(); i ++){
      int [][] crtGroup = dataset.getDataArray()[i];
      Set<Integer> goldPos = dataset.getPositiveLabelsArray()[i];
      
      // zs - all labels with non-zero scores for each datum
      List<Counter<Integer>> zs = estimateZ(crtGroup);
      int [] zPredicted = generateZPredicted(zs);
      // yPredicted - Y labels predicted using the current Zs
      Counter<Integer> yPredicted = estimateY(zPredicted);
      if(verbose) predictionYStats(yPredicted.keySet());
      
      for(Integer l: goldPos) {
        seenLabels.add(l);
        total.incrementCount(l);
        total.incrementCount(LABEL_ALL);
      }
      for(Integer l: yPredicted.keySet()) {
        seenLabels.add(l);
        predicted.incrementCount(l);
        predicted.incrementCount(LABEL_ALL);
        if(goldPos.contains(l)){
          correct.incrementCount(l);
          correct.incrementCount(LABEL_ALL);
        }
      }
    }
    
    Triple<Double, Double, Double> overallScore = computeScore(LABEL_ALL, total, predicted, correct);
    Log.severe("Overall score: P " + overallScore.first() + 
        " R " + overallScore.second() + 
        " F1 " + overallScore.third());
  }
  
  private static Triple<Double, Double, Double> computeScore(int label,
      Counter<Integer> totalCounts,
      Counter<Integer> predictedCounts,
      Counter<Integer> correctCounts) {
    double total = totalCounts.getCount(label);
    double pred = predictedCounts.getCount(label);
    double correct = correctCounts.getCount(label);
    double p = (pred > 0 ? correct / pred : 0.0);
    double r = (total > 0 ? correct / total : 0.0);
    double f1 = (p + r > 0 ? 2 * p * r / (p + r) : 0.0);
    return new Triple<Double, Double, Double>(p, r, f1);
  }
  
  private int [] removeGold(int [] zs, Set<Integer> goldNeg) {
    for(int i = 0; i < zs.length; i ++){
      if(zs[i] != nilIndex && goldNeg.contains(zs[i])){
        zs[i] = nilIndex;
      }
    }
    return zs;
  }
  
  @SuppressWarnings("unchecked")
  private Set<Integer> [] generateZUpdate(
      int [] zUnknown, 
      Set<Integer> goldPos,
      List<Counter<Integer>> zs) {  
    Set<Integer> [] zUpdate = new Set[zs.size()];
    for(int i = 0; i < zUpdate.length; i ++) 
      zUpdate[i] = new HashSet<Integer>();
    
    //
    // A)
    // let's make sure the "at least once" condition is satisfied:
    // for each gold Y pick the highest score Z from the matrix
    //
    // note: this is a greedy algorithm; not guaranteed to be optimal
    // note: by accepting only scores > MIN_VALUE, we are forcing 
    //       the algorithm to start with a local perceptron (step C),
    //       which is more robust during the initial steps
    //
    
    // keeps track of which Y was not mapped to at least a Z
    Set<Integer> unsolvedYs = new HashSet<Integer>(goldPos);
    // keeps track which Zs are already mapped to an Y
    Set<Integer> assignedZs = new HashSet<Integer>();
    while(unsolvedYs.size() > 0 && assignedZs.size() < zs.size()) {
      // loop thru Zs and find the largest assignment to a gold Y
      int bestPos = -1;
      double bestScore = Double.MIN_VALUE;
      int bestY = -1;
      for(int i = 0; i < zs.size(); i ++){
        if(assignedZs.contains(i)) continue;
        for(Integer goldY: unsolvedYs) {
          double score = zs.get(i).getCount(goldY);
          if(score > bestScore) {
            bestScore = score;
            bestPos = i;
            bestY = goldY;
          }
        }
      }
      
      if(bestPos == -1) break;
      zUpdate[bestPos].add(bestY);
      assignedZs.add(bestPos);
      unsolvedYs.remove(bestY);
    }

    //
    // B)
    // first, map all unassigned Zs to an unknown Y, if it was the top prediction
    // second: map the remaining Zs to their top gold label
    // note: these Zs are still kept as unassigned, i.e., they can be reused in (C)
    //
    for(int i = 0; i < zs.size(); i ++){
      // already used in the previous step
      if(assignedZs.contains(i)) continue;
      
      // mapped to an unknown Y; keep that for now
      if(zUnknown != null && zUnknown[i] != nilIndex) {
        zUpdate[i].add(zUnknown[i]);
        continue;
      }
      
      // find the best gold Y for this Z
      List<Pair<Integer, Double>> sortedPredictions = sortPredictions(zs.get(i));
      for(Pair<Integer, Double> pred: sortedPredictions) {
        if(goldPos.contains(pred.first())) {
          zUpdate[i].add(pred.first()); 
          unsolvedYs.remove(pred.first()); 
          break;
        }
      }
    }
    
    //
    // C)
    // map all unsolved Ys to all unassigned Zs
    // this backs off to a local perceptron
    //
    if(unsolvedYs.size() > 0){
      for(int i = 0; i < zs.size(); i ++){
        // used by "at least once"; cannot reuse here
        if(assignedZs.contains(i)) continue;
        
        for(int y: unsolvedYs) {
          zUpdate[i].add(y);
        }
      }
    }
    
    return zUpdate;
  }
  
  private static String arrayToString(Set<Integer> [] array) {
    StringBuilder os = new StringBuilder();
    for(int i = 0; i < array.length; i ++){
      if(i > 0) os.append(" ");
      os.append(array[i]);
    }
    return os.toString();
  }
  
  private static String arrayToString(int [] array) {
    StringBuilder os = new StringBuilder();
    for(int i = 0; i < array.length; i ++){
      if(i > 0) os.append(" ");
      os.append(array[i]);
    }
    return os.toString();
  }
  
  private void inputStats(int index,
      Set<Integer> goldPos,
      Set<Integer> goldNeg,
      int [][] datums) {
    System.err.println("Group #" + index + " with " + datums.length + " datums.");
    System.err.println("y+: " + goldPos);
    System.err.println("y-: " + goldNeg);
    for(int i = 0; i < datums.length; i ++){
      System.err.print("Datum #" + i + ":");
      for(int j = 0; j < datums[i].length; j ++){
        System.err.print(" " + datums[i][j]);
      }
      System.err.println();
    }
    System.err.println();
  }
  
  private void predictionZStats(int [] zs) {
    System.err.print("Predicted z:");
    for(int z: zs)
      System.err.print(" " + z);
    System.err.println("\n");
  }
  
  private void predictionYStats(Set<Integer> ys) {
    System.err.println("Predicted y: " + ys);
  }
  
  private List<Counter<Integer>> estimateZ(int [][] datums) {
    List<Counter<Integer>> zs = new ArrayList<Counter<Integer>>();
    for(int [] datum: datums) {
      zs.add(estimateZ(datum));
    }
    return zs;
  }
  
  private Counter<Integer> estimateZ(int [] datum) {
    Counter<Integer> vector = new ClassicCounter<Integer>();
    for(int d: datum) vector.incrementCount(d);
    
    Counter<Integer> scores = new ClassicCounter<Integer>();
    for(int label = 0; label < zWeights.length; label ++){
      double score = zWeights[label].dotProduct(vector);
      if(score > 0) {
        // only store labels that received a non-zero score
        scores.setCount(label, score);
      }
    }
    
    return scores;
  }
  
  private static int pickBestLabel(Counter<Integer> scores) {
    assert(scores.size() > 0);
    List<Pair<Integer, Double>> sortedScores = sortPredictions(scores);
    return sortedScores.iterator().next().first();
  }
  
  private int [] generateZPredicted(List<Counter<Integer>> zs) {
    int [] bestZs = new int[zs.size()];
    
    for(int i = 0; i < zs.size(); i ++) {
      Counter<Integer> cands = zs.get(i);
      int bestZ = nilIndex;
      if(cands.size() > 0){
        bestZ = pickBestLabel(cands);
      }
      bestZs[i] = bestZ;
    }
    
    return bestZs;
  }
  
  private Counter<Integer> estimateY(int [] zPredicted) {
    return deterministicEstimateY(zPredicted);    
  }
  
  private Counter<Integer> deterministicEstimateY(int [] zPredicted) {  
    Counter<Integer> ys = new ClassicCounter<Integer>();
    for(int zp: zPredicted) {
      if(zp != nilIndex) {
        ys.setCount(zp, 1);
      }
    }
    return ys;
  }
  
  private static Set<Integer> computeUpdateY(Set<Integer> y, Set<Integer> yPos, Set<Integer> yNeg) {
    Set<Integer> yUpdate = new HashSet<Integer>();
    
    for(Integer l: yPos){
      if(! y.contains(l) || ! yNeg.contains(l)) {
        yUpdate.add(l);
      }
    }
    
    for(Integer l: y) {
      if(! yNeg.contains(l)) {
        yUpdate.add(l);
      }
    }
    
    return yUpdate;
  }
  
  private static boolean updateCondition(Set<Integer> y, Set<Integer> yPos, Set<Integer> yNeg) {
    for(Integer l: yPos) {
      if(! y.contains(l)) {
        return true;
      }
    }
    
    for(Integer l: yNeg) {
      if(y.contains(l)) {
        return true;
      }
    }
    
    return false;
  }
  
  private static final int LABEL_ALL = -1;
  
  private void updateZModel(
      Set<Integer> [] goldZ, 
      int [] predictedZ, 
      int [][] group,
      Set<Integer> unknownY,
      List<Counter<Integer>> zProbabilities) {
    assert(goldZ.length == group.length);
    assert(predictedZ.length == group.length);
    
    if(verbose) System.err.println("Updating model using:\n\tgoldZ = " 
        + arrayToString(goldZ) + "\n\tpredZ = " + arrayToString(predictedZ));
    
    for(int i = 0; i < group.length; i ++) {
      Set<Integer> gold = goldZ[i];
      int pred = predictedZ[i];
      int [] datum = group[i];
      
      // negative update
      if(pred != nilIndex && ! gold.contains(pred)) {
        zWeights[pred].update(datum, -1.0);
        if(verbose) System.err.println("Update --- on label " + pred);
        negUpdateStats.incrementCount(pred);
        negUpdateStats.incrementCount(LABEL_ALL);
      }
      
      // negative update for NIL
      if(pred == nilIndex && gold.size() != 0){
        zWeights[nilIndex].update(datum, -1.0);
        if(verbose) System.err.println("Update --- on label NIL");
        negUpdateStats.incrementCount(pred);
      }
      
      // positive + negative update for unknown Ys
      if(SOFT_UNKNOWN && unknownY != null && unknownY.contains(pred)){
        double prob = zProbabilities.get(i).getCount(pred);
        // System.err.println("UNKNOWN_PROB " + pred + " " + prob);
        zWeights[pred].update(datum, /* subsampleRatio * */ prob - 1.0);
        unknownUpdateStats.incrementCount(pred);
        unknownUpdateStats.incrementCount(LABEL_ALL);
      }
      
      // positive update(s)
      for(int l: gold) {
        if(l != nilIndex && l != pred) {
          zWeights[l].update(datum, +1.0);
          if(verbose) System.err.println("Update +++ on label " + l);
          posUpdateStats.incrementCount(l);
          posUpdateStats.incrementCount(LABEL_ALL);
        }
      }      
      
      // positive update for NIL
      if(gold.size() == 0 && pred != nilIndex){
        zWeights[nilIndex].update(datum, +1.0);
        if(verbose) System.err.println("Update +++ on label NIL");
        posUpdateStats.incrementCount(nilIndex);
      }
    }
  }
  
  @Override
  public Counter<String> classifyMentions(List<Collection<String>> sentences) {
    String[] zLabels = new String[sentences.size()];
    Counter<String> [] zLogProbs = 
      ErasureUtils.<Counter<String> []>uncheckedCast(new Counter[sentences.size()]);
      
    //
    // Z level predictions
    //
    Counter<String> localNoisyOr = new ClassicCounter<String>();
    Counter<String> localBest = new ClassicCounter<String>();
    for (int i = 0; i < sentences.size(); i++) {
      Collection<String> sentence = sentences.get(i);
      Counter<String> probs = classifyLocally(sentence);
      
      zLogProbs[i] = new ClassicCounter<String>();
      for(String l: probs.keySet()) {
        zLogProbs[i].setCount(l, Math.log(probs.getCount(l)));
      }
      
      List<Pair<String, Double>> sortedProbs = JointBayesRelationExtractor.sortPredictions(probs);
      Pair<String, Double> prediction = sortedProbs.get(0);
      String l = prediction.first();
      double s = prediction.second();
      zLabels[i] = l;
      // we do not output NIL labels
      if(! zLabels[i].equals(RelationMention.UNRELATED)) {
        double crt = (localNoisyOr.containsKey(l) ? localNoisyOr.getCount(l) : 1.0);
        crt = crt * (1.0 - s);
        localNoisyOr.setCount(l, crt);
        
        double prev = localBest.getCount(l);
        if(s > prev) {
          localBest.setCount(l, s);
        }
      }
    }
    
    //
    // Y level predictions
    // we assign to each predicted label a score equal to the noisy or of the local probabilities
    //
    Counter<String> joint = new ClassicCounter<String>();
    for(String y: localNoisyOr.keySet()) {
      double zProb = (1.0 - localNoisyOr.getCount(y)); 
      joint.setCount(y, zProb);
    }
    return joint;
    // return localBest;
  }
  
  private static List<Pair<Integer, Double>> sortPredictions(Counter<Integer> scores) {
    List<Pair<Integer, Double>> sortedScores = new ArrayList<Pair<Integer,Double>>();
    for(Integer key: scores.keySet()) {
      sortedScores.add(new Pair<Integer, Double>(key, scores.getCount(key)));
    }
    sortPredictions(sortedScores);
    return sortedScores;
  }
  
  private static void sortPredictions(List<Pair<Integer, Double>> scores) {
    Collections.sort(scores, new Comparator<Pair<Integer, Double>>() {
      @Override
      public int compare(Pair<Integer, Double> o1, Pair<Integer, Double> o2) {
        if(o1.second() > o2.second()) return -1;
        if(o1.second().equals(o2.second())) {
          // this is an arbitrary decision to disambiguate ties
          if(o1.first() > o2.first()) return -1;
          else if(o1.first().equals(o2.first())) return 0;
          return 1;
        }
        return 1;
      }
    });
  }
  
  private Counter<String> classifyLocally(Collection<String> testDatum) {
    // fetch all scores 
    Counter<Integer> allLabelScores = new ClassicCounter<Integer>();
    // stores all scores also here; needed for the softmax normalization
    List<Double> scores = new ArrayList<Double>();
    // scan all labels; this includes NIL, which is needed for proper softmax
    for(int labelIdx = 0; labelIdx < zWeights.length; labelIdx ++){
      double score = zWeights[labelIdx].avgDotProduct(testDatum, zFeatureIndex);
      allLabelScores.setCount(labelIdx, score);
      scores.add(score);
    }
    
    // convert scores to probabilities using softmax
    Counter<String> result = new ClassicCounter<String>();
    for(Integer z: allLabelScores.keySet()){
      String l = labelIndex.get(z);
      result.setCount(l, Softmax.softmax(allLabelScores.getCount(z), scores, gamma));
    }

    return result;
  }
}
