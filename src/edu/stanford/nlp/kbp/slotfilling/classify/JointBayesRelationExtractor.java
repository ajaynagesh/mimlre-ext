package edu.stanford.nlp.kbp.slotfilling.classify;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

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
import java.util.Random;
import java.util.Set;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.classify.WeightedDataset;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.PropertiesUtils;

/**
 * Implements the MIML-RE model from EMNLP 2012
 * @author Julie Tibshirani (jtibs)
 * @author nallapat@ai.sri.com
 * @author Mihai
 *
 */
public class JointBayesRelationExtractor extends JointlyTrainedRelationExtractor{
  private static final long serialVersionUID = -7961154075748697901L;
  
  static enum LOCAL_CLASSIFICATION_MODE {
    WEIGHTED_VOTE,
    SINGLE_MODEL
  }
  
  static enum InferenceType {
    SLOW,
    STABLE
  }

  private static final LOCAL_CLASSIFICATION_MODE localClassificationMode = 
    LOCAL_CLASSIFICATION_MODE.WEIGHTED_VOTE;
  
  /**
   * sentence-level multi-class classifier, trained across all sentences 
   * one per fold to avoid overfitting
   */
  private LinearClassifier<String, String> [] zClassifiers;
  /** this is created only if localClassificationMode == SINGLE_MODEL */
  LinearClassifier<String, String> zSingleClassifier;
  /** one two-class classifier for each top-level relation */
  private Map<String, LinearClassifier<String, String>> yClassifiers;
  
  private Index<String> featureIndex;
  private Index<String> yLabelIndex;
  private Index<String> zLabelIndex;
  
  private static String ATLEASTONCE_FEAT = "atleastonce";
  private static String NONE_FEAT = "none";
  private static List<String> Y_FEATURES_FOR_INITIAL_MODEL;
  
  static {
    Y_FEATURES_FOR_INITIAL_MODEL = new ArrayList<String>();
    Y_FEATURES_FOR_INITIAL_MODEL.add(ATLEASTONCE_FEAT);
    Y_FEATURES_FOR_INITIAL_MODEL.add(NONE_FEAT);
  }
  
  /** Run EM for this many epochs */
  private final int numberOfTrainEpochs;
  
  /** Organize the Z classifiers into this many folds for cross validation */
  private int numberOfFolds;
  
  /** 
   * Should we skip the EM loop?
   * If true, this is essentially a multiclass local LR classifier
   */
  private final boolean onlyLocalTraining;
  
  /** Required to know where to save the initial models */
  private final String initialModelPath;
  
  /** Sigma for the Z classifiers */
  private final double zSigma;
  /** Sigma for the Y classifiers */
  private final double ySigma;
  
  /** Counts number of flips for Z labels in one epoch */
  private int zUpdatesInOneEpoch = 0;
  
  private final LocalFilter localDataFilter;
  
  private final InferenceType inferenceType;
  
  /** Which feature model to use */
  private final int featureModel;
  
  /** Should we train Y models? */
  private final boolean trainY;
  
  /** These label dependencies were seen in training */
  private Set<String> knownDependencies;
  
  private String serializedModelPath;
    
  public JointBayesRelationExtractor(Properties props) {
    this(props, false);
  }
  
  public JointBayesRelationExtractor(Properties props, boolean onlyLocal) {
    // We need workDir, serializedRelationExtractorName, modelType, samplingRatio to serialize the initial models
    String workDir = props.getProperty(Props.WORK_DIR);
    String srn = props.getProperty(Props.SERIALIZED_MODEL_PATH, "kbp_relation_model");
    if (srn.endsWith(Constants.SER_EXT))
      srn = srn.substring(0, srn.length() - Constants.SER_EXT.length());
    String serializedRelationExtractorName = srn;
    ModelType modelType = ModelType.stringToModel(props.getProperty(Props.MODEL_TYPE, Constants.DEFAULT_MODEL));
    double samplingRatio = PropertiesUtils.getDouble(props, 
        Props.NEGATIVES_SAMPLE_RATIO,
        Constants.DEFAULT_NEGATIVES_SAMPLING_RATIO);
    initialModelPath = makeInitialModelPath(
        workDir, 
        serializedRelationExtractorName, 
        modelType, 
        samplingRatio);
    numberOfTrainEpochs = PropertiesUtils.getInt(props,
        Props.EPOCHS, 10);
    numberOfFolds = PropertiesUtils.getInt(props,
        Props.FOLDS, 5);
    zSigma = 1.0;
    ySigma = 1.0;
    localDataFilter = 
      makeLocalDataFilter(props.getProperty(
        Props.FILTER, "all"));
    inferenceType =
      makeInferenceType(props.getProperty(
          Props.INFERENCE_TYPE, "stable"));
    featureModel = PropertiesUtils.getInt(props,
        Props.FEATURES, 0);
    trainY = PropertiesUtils.getBool(props,
        Props.TRAINY, true);
    onlyLocalTraining = onlyLocal;
    serializedModelPath = makeModelPath(
        workDir, 
        serializedRelationExtractorName, 
        modelType, 
        samplingRatio);
  }
  
  private static InferenceType makeInferenceType(String v) {
    if(v.equalsIgnoreCase("slow")) return InferenceType.SLOW;
    if(v.equalsIgnoreCase("stable")) return InferenceType.STABLE;
    throw new RuntimeException("ERROR: unknown inference type " + v);
  }
  
  private static LocalFilter makeLocalDataFilter(String fv) {
    LocalFilter localDataFilter;
    if(fv.equalsIgnoreCase("all")) 
      localDataFilter = new AllFilter();
    else if(fv.equalsIgnoreCase("single")) 
      localDataFilter = new SingleFilter();
    else if(fv.equals("redundancy"))
      localDataFilter = new RedundancyFilter();
    else if(fv.startsWith("large")) {
      int thr = Integer.valueOf(fv.substring(5));
      assert(thr > 0);
      localDataFilter = new LargeFilter(thr);
    }
    else
      throw new RuntimeException("ERROR: unknown local data filter " + fv);
    Log.severe("Using local data filter: " + fv);
    return localDataFilter;
  }
  
  public JointBayesRelationExtractor(
      String initialModelPath,
      int numberOfTrainEpochs,
      int numberOfFolds,
      String localFilter, 
      int featureModel,
      String inferenceType,
      boolean trainY,
      boolean onlyLocalTraining) {
    this.initialModelPath = initialModelPath;
    this.numberOfTrainEpochs = numberOfTrainEpochs;
    this.numberOfFolds = numberOfFolds;
    this.zSigma = 1.0;
    this.ySigma = 1.0;
    this.onlyLocalTraining = onlyLocalTraining;
    this.localDataFilter = makeLocalDataFilter(localFilter);
    this.featureModel = featureModel;
    this.inferenceType = makeInferenceType(inferenceType);
    this.trainY = trainY;
    this.serializedModelPath = null;
  }
  
  public void setSerializedModelPath(String p) {
    serializedModelPath = p;
  }
  
  private static String makeInitialModelPath(
      String workDir, 
      String serializedRelationExtractorName, 
      ModelType modelType, 
      double samplingRatio) {
    return workDir + File.separator + serializedRelationExtractorName + 
      "." + modelType + "." + (int) (100.0 * samplingRatio) + 
      ".initial" + Constants.SER_EXT;
  }
  private static String makeModelPath(
      String workDir, 
      String serializedRelationExtractorName, 
      ModelType modelType, 
      double samplingRatio) {
    return workDir + File.separator + serializedRelationExtractorName + 
      "." + modelType + "." + (int) (100.0 * samplingRatio) + 
      Constants.SER_EXT;
  }
  
  private int foldStart(int fold, int size) {
    int foldSize = size / numberOfFolds;
    assert(foldSize > 0);
    int start = fold * foldSize;
    assert(start < size);
    return start;
  }
  
  private int foldEnd(int fold, int size) {
    // padding if this is the last fold
    if(fold == numberOfFolds - 1) 
      return size;
    
    int foldSize = size / numberOfFolds;
    assert(foldSize > 0);
    int end = (fold + 1) * foldSize;
    assert(end <= size);
    return end;
  }
  
  private int [][] initializeZLabels(MultiLabelDataset<String, String> data) {
    // initialize Z labels with the predictions of the local classifiers
    int[][] zLabels = new int[data.getDataArray().length][];
    for(int f = 0; f < numberOfFolds; f ++){
      LinearClassifier<String, String> zClassifier = zClassifiers[f];
      assert(zClassifier != null);
      for(int i = foldStart(f, data.getDataArray().length); i < foldEnd(f, data.getDataArray().length); i ++){
        int [][] group = data.getDataArray()[i];
        zLabels[i] = new int[group.length];
        for(int j = 0; j < group.length; j ++){
          int [] datum = group[j];
          Counter<String> scores = zClassifier.scoresOf(datum);
          List<Pair<String, Double>> sortedScores = sortPredictions(scores);
          int sys = zLabelIndex.indexOf(sortedScores.get(0).first());
          assert(sys != -1);
          zLabels[i][j] = sys;
        }
      }
    }
    
    return zLabels;
  }
  
  private void detectDependencyYFeatures(MultiLabelDataset<String, String> data) {
    knownDependencies = new HashSet<String>();
    for(int i = 0; i < data.size(); i ++){
      Set<Integer> labels = data.getPositiveLabelsArray()[i];
      for(Integer src: labels) {
        String srcLabel = data.labelIndex().get(src);
        for(Integer dst: labels) {
          if(src.intValue() == dst.intValue()) continue;
          String dstLabel = data.labelIndex().get(dst);
          String f = makeCoocurrenceFeature(srcLabel, dstLabel);
          Log.severe("FOUND COOC: " + f);
          knownDependencies.add(f);
        }
      }
    }
  }
  
  private static String makeCoocurrenceFeature(String src, String dst) {
    return "co:s|" + src + "|d|" + dst + "|";
  }
  
  @Override
  public void train(MultiLabelDataset<String, String> data) {
    
    // filter some of the groups
    if(localDataFilter instanceof LargeFilter) {
      List<int [][]> filteredGroups = new ArrayList<int[][]>();
      List<Set<Integer>> filteredPosLabels = new ArrayList<Set<Integer>>();
      List<Set<Integer>> filteredNegLabels = new ArrayList<Set<Integer>>();
      for(int i = 0; i < data.size(); i ++) {
        if(localDataFilter.filterY(data.getDataArray()[i], data.getPositiveLabelsArray()[i])) {
          filteredGroups.add(data.getDataArray()[i]);
          filteredPosLabels.add(data.getPositiveLabelsArray()[i]);
          filteredNegLabels.add(data.getNegativeLabelsArray()[i]);
        }
      }
      data = new MultiLabelDataset<String, String>(
          filteredGroups.toArray(new int[filteredGroups.size()][][]),
          data.featureIndex(), data.labelIndex(), 
          filteredPosLabels.toArray(ErasureUtils.<Set<Integer> []>uncheckedCast(new Set[filteredPosLabels.size()])),
          filteredNegLabels.toArray(ErasureUtils.<Set<Integer> []>uncheckedCast(new Set[filteredNegLabels.size()])));
    }
    
    LinearClassifierFactory<String, String> zFactory =
      new LinearClassifierFactory<String, String>(1e-4, false, zSigma);
    LinearClassifierFactory<String, String> yFactory =
      new LinearClassifierFactory<String, String>(1e-4, false, ySigma);
    zFactory.setVerbose(false);
    yFactory.setVerbose(false);
    
    System.out.println("DataSize : " + data.data.length);
    System.out.println("Pos LabelSize : " + data.posLabels.length);
    System.out.println("Neg LabelSize : " + data.negLabels.length);
    if(initialModelPath != null && new File(initialModelPath).exists()) {
      try {
        loadInitialModels(initialModelPath);
        // yClassifiers = initializeYClassifiersWithAtLeastOnce(yLabelIndex);
      } catch (Exception e1) {
        throw new RuntimeException(e1);
      }
    } else {
      featureIndex = data.featureIndex();
      yLabelIndex = data.labelIndex();
      zLabelIndex = new HashIndex<String>(yLabelIndex);
      zLabelIndex.add(RelationMention.UNRELATED);
      
      // initialize classifiers 
      zClassifiers = initializeZClassifierLocally(data, featureIndex, zLabelIndex);
      yClassifiers = initializeYClassifiersWithAtLeastOnce(yLabelIndex);

      if(initialModelPath != null) {
        try {
          saveInitialModels(initialModelPath);
        } catch (IOException e1) {
          throw new RuntimeException(e1);
        }
      }
    }
    
    // stop training after initialization
    // this is essentially a local model!
    if(onlyLocalTraining) return;
    
    detectDependencyYFeatures(data);
    
    for(String y: yLabelIndex) {
      int yi = yLabelIndex.indexOf(y);
      Log.severe("YLABELINDEX " + y + " = " + yi);
    }
    
    // calculate total number of sentences
    int totalSentences = 0;
    for (int[][] group : data.getDataArray())
      totalSentences += group.length;

    // initialize predicted z labels
    int[][] zLabels = initializeZLabels(data);
    computeConfusionMatrixForCounts("LOCAL", zLabels, data.getPositiveLabelsArray());
    computeYScore("LOCAL", zLabels, data.getPositiveLabelsArray());
    
    // z dataset initialized with nil labels and the sentence-level features array
    // Dataset<String, String> zDataset = initializeZDataset(totalSentences, zLabels, data.getDataArray());
    
    // y dataset initialized to be empty, as it will be populated during the E step
    Map<String, RVFDataset<String, String>> yDatasets = initializeYDatasets();
       
    // run EM
    for (int epoch = 0; epoch < numberOfTrainEpochs; epoch++) {
      zUpdatesInOneEpoch = 0;
      Log.severe("***EPOCH " + epoch + "***");

      // we compute scores in each epoch using these labels
      int [][] zLabelsPredictedByZ = new int[zLabels.length][];
      for(int i = 0; i < zLabels.length; i ++)
        zLabelsPredictedByZ[i] = new int[zLabels[i].length];
      
      //
      // E-step
      //
      Log.severe("E-STEP");
      // for each group, infer the hidden sentence labels z_i,s
      for(int fold = 0; fold < numberOfFolds; fold ++) {
        LinearClassifier<String, String> zClassifier = zClassifiers[fold];
        int start = foldStart(fold, data.getDataArray().length);
        int end = foldEnd(fold, data.getDataArray().length);
        
        for (int i = start; i < end; i++) {
          int[][] group = data.getDataArray()[i];
          randomizeGroup(group, epoch);
          
          Set<Integer> positiveLabels = data.getPositiveLabelsArray()[i];
          Set<Integer> negativeLabels = data.getNegativeLabelsArray()[i];
          Counter<String> [] zLogProbs = 
            ErasureUtils.uncheckedCast(new Counter[group.length]);
          
          predictZLabels(group, zLabelsPredictedByZ[i], zClassifier);
            
          switch(inferenceType) {
          case SLOW: 
            inferZLabels(group, positiveLabels, negativeLabels, zLabels[i], zLogProbs, zClassifier, epoch);
            break;
          case STABLE:
            inferZLabelsStable(group, positiveLabels, negativeLabels, zLabels[i], zLogProbs, zClassifier, epoch);
            break;
          default:
            throw new RuntimeException("ERROR: unknown inference type: " + inferenceType);
          }

          // given these predicted z labels, update the features in the y dataset
          //printGroup(zLabels[i], positiveLabels);
          for (int y : positiveLabels) {
            String yLabel = yLabelIndex.get(y);
            addYDatum(yDatasets.get(yLabel), yLabel, zLabels[i], zLogProbs, true);
          }
          for (int y : negativeLabels) {
            String yLabel = yLabelIndex.get(y);
            addYDatum(yDatasets.get(yLabel), yLabel, zLabels[i], zLogProbs, false);
          }
        }
      }
      
      computeConfusionMatrixForCounts("EPOCH " + epoch, zLabels, data.getPositiveLabelsArray());
      computeYScore("EPOCH " + epoch, zLabels, data.getPositiveLabelsArray());
      computeYScore("(Z ONLY) EPOCH " + epoch, zLabelsPredictedByZ, data.getPositiveLabelsArray());
      
      Log.severe("In epoch #" + epoch + " zUpdatesInOneEpoch = " + zUpdatesInOneEpoch);
      if(zUpdatesInOneEpoch == 0){
        Log.severe("Stopping training. Did not find any changes in the Z labels!");
        break;
      }
      
      // update the labels in the z dataset
      Dataset<String, String> zDataset = initializeZDataset(totalSentences, zLabels, data.getDataArray());

      //
      // M step
      // 
      Log.severe("M-STEP");
      // learn the weights of the sentence-level multi-class classifier
      for(int fold = 0; fold < numberOfFolds; fold ++){
        Log.severe("EPOCH " + epoch + ": Training Z classifier for fold #" + fold);
        int [][] foldTrainArray = makeTrainDataArrayForFold(zDataset.getDataArray(), fold);
        int [] foldTrainLabels = makeTrainLabelArrayForFold(zDataset.getLabelsArray(), fold);
        Dataset<String, String> zd = new Dataset<String, String>(zLabelIndex, foldTrainLabels, featureIndex, foldTrainArray);
        LinearClassifier<String, String> zClassifier = zFactory.trainClassifier(zd);
        zClassifiers[fold] = zClassifier;
      }
      
      // learn the weights of each of the top-level two-class classifiers
      if(trainY) {
        for (String yLabel : yLabelIndex) {
          Log.severe("EPOCH " + epoch + ": Training Y classifier for label " + yLabel);
          RVFDataset<String, String> trainSet = yDatasets.get(yLabel);
          yClassifiers.put(yLabel, yFactory.trainClassifier(trainSet));
        }
      }
          
      // save this epoch's model
      String epochPath = makeEpochPath(epoch);
      try {
        if(epochPath != null) {
          makeSingleZClassifier(zDataset, zFactory);
          save(epochPath);
        }
      } catch (IOException ex) {
        Log.severe("WARNING: could not save model of epoch " + epoch + " to path: " + epochPath);
        Log.severe("Exception message: " + ex.getMessage());
      }
      
      // clear our y datasets so they can be repopulated on next iteration
      yDatasets = initializeYDatasets();
    }
    
    Dataset<String, String> zDataset = initializeZDataset(totalSentences, zLabels, data.getDataArray());
    makeSingleZClassifier(zDataset, zFactory);
  }
  
  void randomizeGroup(int[][] group, int randomSeed) {
    Random rand = new Random(randomSeed);
    for(int j = group.length - 1; j > 0; j --){
      int randIndex = rand.nextInt(j);
      
      int [] tmp = group[randIndex];
      group[randIndex] = group[j];
      group[j] = tmp;
    }
  }
  
  void computeYScore(String name, int [][] zLabels, Set<Integer> [] golds) {
    int labelCorrect = 0, labelPredicted = 0, labelTotal = 0;
    int groupCorrect = 0, groupTotal = 0;
    int nilIndex = zLabelIndex.indexOf(RelationMention.UNRELATED);
    for(int i = 0; i < golds.length; i ++) {
      Set<Integer> pred = new HashSet<Integer>();
      for(int z: zLabels[i]) 
        if(z != nilIndex) pred.add(z);
      Set<Integer> gold = golds[i];
      
      labelPredicted += pred.size();
      labelTotal += gold.size();
      for(int z: pred) 
        if(gold.contains(z))
          labelCorrect ++;
      
      groupTotal ++;
      boolean correct = true;
      if(pred.size() != gold.size()) {
        correct = false;
      } else {
        for(int z: pred) { 
          if(! gold.contains(z)) {
            correct = false;
            break;
          }
        }
      }
      if(correct) groupCorrect ++;
    }
    
    double p = (double) labelCorrect / (double) labelPredicted;
    double r = (double) labelCorrect / (double) labelTotal;
    double f1 = p != 0 && r != 0 ? 2*p*r/(p + r) : 0;
    double a = (double) groupCorrect / (double) groupTotal;
    Log.severe("LABEL SCORE for " + name + ": P " + p + " R " + r + " F1 " + f1);
    Log.severe("GROUP SCORE for " + name + ": A " + a);
  }
  
  void computeConfusionMatrixForCounts(String name, int [][] zLabels, Set<Integer> [] golds) {
    Counter<Integer> pos = new ClassicCounter<Integer>();
    Counter<Integer> neg = new ClassicCounter<Integer>();
    int nilIndex = zLabelIndex.indexOf(RelationMention.UNRELATED);
    for(int i = 0; i < zLabels.length; i ++) {
      int [] zs = zLabels[i];
      Counter<Integer> freqs = new ClassicCounter<Integer>();
      for(int z: zs) 
        if(z != nilIndex) 
          freqs.incrementCount(z);
      Set<Integer> gold = golds[i];
      for(int z: freqs.keySet()) {
        int f = (int) freqs.getCount(z);
        if(gold.contains(z)){
          pos.incrementCount(f);
        } else {
          neg.incrementCount(f);
        }
      }
    }
    Log.severe("CONFUSION MATRIX for " + name);
    Log.severe("CONFUSION MATRIX POS: " + pos);
    Log.severe("CONFUSION MATRIX NEG: " + neg);
  }

  void printGroup(int [] zs, Set<Integer> ys) {
    System.err.print("ZS:");
    for(int z: zs) {
      String zl = zLabelIndex.get(z);
      System.err.print(" " + zl);
    }
    System.err.println();
    Set<String> missed = new HashSet<String>();
    System.err.print("YS:");
    for(Integer y: ys) {
      String yl = yLabelIndex.get(y);
      System.err.print(" " + yl);
      boolean found = false;
      for(int z: zs) {
        String zl = zLabelIndex.get(z);
        if(zl.equals(yl)) {
          found = true;
          break;
        }
      }
      if(! found) {
        missed.add(yl);
      }
    }
    System.err.println();
    if(missed.size() > 0) {
      System.err.print("MISSED " + missed.size() + ":");
      for(String m: missed) {
        System.err.print(" " + m);
      }
    }
    System.err.println();
    System.err.println("END GROUP");
  }

  private void addYDatum(
      RVFDataset<String, String> yDataset,
      String yLabel, 
      int [] zLabels,
      Counter<String> [] zLogProbs,
      boolean isPositive) {
    Counter<String> yFeats = extractYFeatures(yLabel, zLabels, zLogProbs);
    //if(yFeats.size() > 0) System.err.println("YFEATS" + (isPositive ? " POSITIVE " : " NEGATIVE ") + yLabel + " " + yFeats);
    RVFDatum<String, String> datum = 
      new RVFDatum<String, String>(yFeats, 
          (isPositive ? yLabel : RelationMention.UNRELATED));
    yDataset.add(datum);
  }
  
  private String makeEpochPath(int epoch) {
    String epochPath = null;
    if(epoch < numberOfTrainEpochs && serializedModelPath != null) {
      if(serializedModelPath.endsWith(".ser")) {
        epochPath = 
          serializedModelPath.substring(0, serializedModelPath.length() - ".ser".length()) +
          "_EPOCH" + epoch + ".ser";
      } else {
        epochPath = serializedModelPath + "_EPOCH" + epoch;
      }
    }
    return epochPath;
  }
  
  private void makeSingleZClassifier(
      Dataset<String, String> zDataset, 
      LinearClassifierFactory<String, String> zFactory) {
    if(localClassificationMode == LOCAL_CLASSIFICATION_MODE.SINGLE_MODEL) {
      // train a single Z classifier over the entire data
      Log.severe("Training the final Z classifier...");
      zSingleClassifier = zFactory.trainClassifier(zDataset);
    } else {
      zSingleClassifier = null;
    }
  }

  private int[] flatten(int[][] array, int size) {
    int[] result = new int[size];
    int count = 0;
    for (int[] row : array) {
      for (int element : row)
        result[count++] = element;
    }
    return result;
  }
  
  static abstract class LocalFilter {
    public abstract boolean filterZ(int [][] data, Set<Integer> posLabels);
    public boolean filterY(int [][] data, Set<Integer> posLabels) { return true; }
  }
  static class LargeFilter extends LocalFilter {
    final int threshold;
    public LargeFilter(int thresh) {
      this.threshold = thresh;
    }
    @Override
    public boolean filterZ(int[][] data, Set<Integer> posLabels) {
      if(data.length > threshold) return false;
      return true;
    }
    @Override
    public boolean filterY(int[][] data, Set<Integer> posLabels) {
      if(data.length > threshold) return false;
      return true;
    }
  }
  static class AllFilter extends LocalFilter {
    @Override
    public boolean filterZ(int[][] data, Set<Integer> posLabels) {
      return true;
    }
  }
  static class SingleFilter extends LocalFilter {
    @Override
    public boolean filterZ(int[][] data, Set<Integer> posLabels) {
      if(posLabels.size() <= 1) return true;
      return false;
    }
  }
  static class RedundancyFilter extends LocalFilter {
    @Override
    public boolean filterZ(int[][] data, Set<Integer> posLabels) {
      if(posLabels.size() <= 1 && data.length > 1) return true;
      return false;
    }
  }
  
  private static Dataset<String, String> makeLocalData(
      int [][][] dataArray, 
      Set<Integer> [] posLabels,
      Index<String> labelIndex,
      Index<String> featureIndex,
      LocalFilter f,
      int fold) {
    // Detect the size of the dataset for the local classifier
    int flatSize = 0, posGroups = 0, negGroups = 0;
    for(int i = 0; i < dataArray.length; i ++) {
      if(! f.filterZ(dataArray[i], posLabels[i])) continue;
      if(posLabels[i].size() == 0) {
        // negative example
        flatSize += dataArray[i].length;
        negGroups ++;
      } else {
        // 1+ positive labels
        flatSize += dataArray[i].length * posLabels[i].size();
        posGroups ++;
      }
    }
    Log.severe("Explored " + posGroups + " positive groups and " + negGroups + " negative groups, yielding " + flatSize + " flat/local datums.");

    //
    // Construct the flat local classifier
    //
    int [][] localTrainData = new int[flatSize][];
    int [] localTrainLabels = new int[flatSize];
    float [] weights = new float[flatSize];
    int offset = 0, posCount = 0;
    Set<Integer> negLabels = new HashSet<Integer>();
    int nilIndex = labelIndex.indexOf(RelationMention.UNRELATED);
    negLabels.add(nilIndex);
    for(int i = 0; i < dataArray.length; i ++) {
      if(! f.filterZ(dataArray[i], posLabels[i])) continue;
      int [][] group = dataArray[i];
      Set<Integer> labels = posLabels[i];
      if(labels.size() == 0) labels = negLabels;
      float weight = (float) 1.0 / (float) labels.size();
      for(Integer label: labels) {
        for(int j = 0; j < group.length; j ++){
          localTrainData[offset] = group[j];
          localTrainLabels[offset] = label;
          weights[offset] = weight;
          if(label != nilIndex) posCount ++;
          offset ++;
          if(offset >= flatSize) break;
        }
        if(offset >= flatSize) break;
      }
      if(offset >= flatSize) break;
    }
    
    Dataset<String, String> dataset = new WeightedDataset<String, String>(
        labelIndex, 
        localTrainLabels, 
        featureIndex, 
        localTrainData, localTrainData.length, 
        weights);
    Log.severe("Fold #" + fold + ": Constructed a dataset with " + localTrainData.length + 
        " datums, out of which " + posCount + " are positive.");
    if(posCount == 0) throw new RuntimeException("ERROR: cannot handle a dataset with 0 positive examples!");
    
    return dataset;
  }
  
  private int [] makeTrainLabelArrayForFold(int [] labelArray, int fold) {
    int start = foldStart(fold, labelArray.length);
    int end = foldEnd(fold, labelArray.length);
    int [] train = new int[labelArray.length - end + start];
    int trainOffset = 0;
    for(int i = 0; i < start; i ++){
      train[trainOffset] = labelArray[i];
      trainOffset ++;
    }
    for(int i = end; i < labelArray.length; i ++){
      train[trainOffset] = labelArray[i];
      trainOffset ++;
    }
    return train;
  }
  
  private int [][] makeTrainDataArrayForFold(int [][] dataArray, int fold) {
    int start = foldStart(fold, dataArray.length);
    int end = foldEnd(fold, dataArray.length);
    int [][] train = new int[dataArray.length - end + start][];
    int trainOffset = 0;
    for(int i = 0; i < start; i ++){
      train[trainOffset] = dataArray[i];
      trainOffset ++;
    }
    for(int i = end; i < dataArray.length; i ++){
      train[trainOffset] = dataArray[i];
      trainOffset ++;
    }
    return train;
  }

  private Pair<int [][][], int [][][]> makeDataArraysForFold(int [][][] dataArray, int fold) {
    int start = foldStart(fold, dataArray.length);
    int end = foldEnd(fold, dataArray.length);
    int [][][] train = new int[dataArray.length - end + start][][];
    int [][][] test = new int[end - start][][];
    int trainOffset = 0, testOffset = 0;
    for(int i = 0; i < dataArray.length; i ++){
      if(i < start){
        train[trainOffset] = dataArray[i];
        trainOffset ++;
      } else if(i < end) {
        test[testOffset] = dataArray[i];
        testOffset ++;
      } else {
        train[trainOffset] = dataArray[i];
        trainOffset ++;
      }
    }
    return new Pair<int[][][], int[][][]>(train, test);
  }
  
  @SuppressWarnings("unchecked")
  private Pair<Set<Integer> [], Set<Integer> []> makeLabelSetsForFold(Set<Integer> [] labelSet, int fold) {
    int start = foldStart(fold, labelSet.length);
    int end = foldEnd(fold, labelSet.length);
    Set<Integer>[] train = new HashSet[labelSet.length - end + start];
    Set<Integer>[] test = new HashSet[end - start];
    int trainOffset = 0, testOffset = 0;
    for(int i = 0; i < labelSet.length; i ++){
      if(i < start){
        train[trainOffset] = labelSet[i];
        trainOffset ++;
      } else if(i < end) {
        test[testOffset] = labelSet[i];
        testOffset ++;
      } else {
        train[trainOffset] = labelSet[i];
        trainOffset ++;
      }
    }
    return new Pair<Set<Integer>[], Set<Integer>[]>(train, test);
  }
  
  @SuppressWarnings("unchecked")
  private LinearClassifier<String, String> [] initializeZClassifierLocally(
      MultiLabelDataset<String, String> data,
      Index<String> featureIndex,
      Index<String> labelIndex) {
    
    LinearClassifier<String, String> [] localClassifiers = new LinearClassifier[numberOfFolds];
    
    // construct the initial model for each fold
    for(int fold = 0; fold < numberOfFolds; fold ++){
      Log.severe("Constructing dataset for the local model in fold #" + fold + "...");
      Pair<int [][][], int [][][]> dataArrays = makeDataArraysForFold(data.getDataArray(), fold);
      Pair<Set<Integer> [], Set<Integer> []> labelSets = 
        makeLabelSetsForFold(data.getPositiveLabelsArray(), fold);
      
      int [][][] trainDataArray = dataArrays.first();
      Set<Integer> [] trainPosLabels = labelSets.first();
      assert(trainDataArray.length == trainPosLabels.length);
      int [][][] testDataArray = dataArrays.second();
      Set<Integer> [] testPosLabels = labelSets.second();
      assert(testDataArray.length == testPosLabels.length);
      
      //
      // Construct the flat local classifier
      //
      Dataset<String, String> dataset = 
        makeLocalData(trainDataArray, trainPosLabels, labelIndex, featureIndex, localDataFilter, fold);

      //
      // Train local classifier
      //
      Log.severe("Fold #" + fold + ": Training local model...");
      LinearClassifierFactory<String, String> factory =
        new LinearClassifierFactory<String, String>(1e-4, false, zSigma);
      LinearClassifier<String, String> localClassifier = factory.trainClassifier(dataset);
      Log.severe("Fold #" + fold + ": Training of the local classifier completed.");

      //
      // Evaluate the classifier on the multidataset
      //
      int nilIndex = labelIndex.indexOf(RelationMention.UNRELATED);
      Log.severe("Fold #" + fold + ": Evaluating the local classifier on the hierarchical dataset...");
      int total = 0, predicted = 0, correct = 0;
      for(int i = 0; i < testDataArray.length; i ++){
        int [][] group = testDataArray[i];
        Set<Integer> gold = testPosLabels[i];
        Set<Integer> pred = new HashSet<Integer>();
        for(int j = 0; j < group.length; j ++){
          int [] datum = group[j];
          Counter<String> scores = localClassifier.scoresOf(datum);
          List<Pair<String, Double>> sortedScores = sortPredictions(scores);
          int sys = labelIndex.indexOf(sortedScores.get(0).first());
          if(sys != nilIndex) pred.add(sys);
        }
        total += gold.size();
        predicted += pred.size();
        for(Integer pv: pred) {
          if(gold.contains(pv)) correct ++;
        }
      }
      double p = (double) correct / (double) predicted;
      double r = (double) correct / (double) total;
      double f1 = (p != 0 && r != 0 ? 2*p*r/(p+r) : 0);
      Log.severe("Fold #" + fold + ": Training score on the hierarchical dataset: P " + p + " R " + r + " F1 " + f1);

      Log.severe("Fold #" + fold + ": Created the Z classifier with " + labelIndex.size() + 
          " labels and " + featureIndex.size() + " features.");
      localClassifiers[fold] = localClassifier;
    }
    
    return localClassifiers;
  }
  
  @SuppressWarnings("unchecked")
  private void loadInitialModels(String path) throws IOException, ClassNotFoundException {
    InputStream is = new FileInputStream(path);
    ObjectInputStream in = new ObjectInputStream(is);
    
    featureIndex = ErasureUtils.uncheckedCast(in.readObject());
    zLabelIndex = ErasureUtils.uncheckedCast(in.readObject());
    yLabelIndex = ErasureUtils.uncheckedCast(in.readObject());
    
    numberOfFolds = in.readInt();
    zClassifiers = new LinearClassifier[numberOfFolds];
    for(int i = 0; i < numberOfFolds; i ++){
      LinearClassifier<String, String> classifier =
        ErasureUtils.uncheckedCast(in.readObject());
      zClassifiers[i] = classifier;
      Log.severe("Loaded Z classifier for fold #" + i + ": " + zClassifiers[i]);
    }
    
    int numLabels = in.readInt();
    yClassifiers = new HashMap<String, LinearClassifier<String, String>>();
    for (int i = 0; i < numLabels; i++) {
      String yLabel = ErasureUtils.uncheckedCast(in.readObject());
      LinearClassifier<String, String> classifier = 
        ErasureUtils.uncheckedCast(in.readObject());
      yClassifiers.put(yLabel, classifier);
    }
    
    in.close();
  }
  
  private void saveInitialModels(String path) throws IOException {
    ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path));
    out.writeObject(featureIndex);
    out.writeObject(zLabelIndex);
    out.writeObject(yLabelIndex);
    out.writeInt(zClassifiers.length);
    for(int i = 0; i < zClassifiers.length; i ++)
      out.writeObject(zClassifiers[i]);
    out.writeInt(yClassifiers.keySet().size());
    for (String yLabel : yClassifiers.keySet()) {
      out.writeObject(yLabel);
      out.writeObject(yClassifiers.get(yLabel));
    }
    out.close();
  }
  
  private Map<String, LinearClassifier<String, String>> initializeYClassifiersWithAtLeastOnce(Index<String> labelIndex) {
    Map<String, LinearClassifier<String, String>> classifiers = 
      new HashMap<String, LinearClassifier<String, String>>();
    for (String yLabel : labelIndex) {
      Index<String> yFeatureIndex = new HashIndex<String>();
      yFeatureIndex.addAll(Y_FEATURES_FOR_INITIAL_MODEL);

      Index<String> thisYLabelIndex = new HashIndex<String>();
      thisYLabelIndex.add(yLabel);
      thisYLabelIndex.add(RelationMention.UNRELATED);

      double[][] weights = initializeWeights(yFeatureIndex.size(), thisYLabelIndex.size());
      setYWeightsForAtLeastOnce(weights, yFeatureIndex, thisYLabelIndex);
      classifiers.put(yLabel, new LinearClassifier<String, String>(weights, yFeatureIndex, thisYLabelIndex));
      Log.severe("Created the classifier for Y=" + yLabel + " with " + yFeatureIndex.size() + " features");
    }
    return classifiers;
  }
  
  private static final double BIG_WEIGHT = +10;
  
  private static void setYWeightsForAtLeastOnce(double[][] weights,
      Index<String> featureIndex,
      Index<String> labelIndex) {
    int posLabel = -1, negLabel = -1;
    for(String l: labelIndex) {
      if(l.equalsIgnoreCase(RelationMention.UNRELATED)) {
        negLabel = labelIndex.indexOf(l);
      } else {
        Log.fine("posLabel = " + l);
        posLabel = labelIndex.indexOf(l);
      }
    }
    assert(posLabel != -1);
    assert(negLabel != -1);
    
    int atLeastOnceIndex = featureIndex.indexOf(ATLEASTONCE_FEAT);
    int noneIndex = featureIndex.indexOf(NONE_FEAT);
    weights[atLeastOnceIndex][posLabel] = BIG_WEIGHT;
    weights[noneIndex][negLabel] = BIG_WEIGHT;
    Log.fine("posLabel = " + posLabel + ", negLabel = " + negLabel + ", atLeastOnceIndex = " + atLeastOnceIndex);
  }
  
  private static double[][] initializeWeights(int numFeatures, int numLabels) {
    double[][] weights = new double[numFeatures][numLabels];
    for (double[] row : weights)
      Arrays.fill(row, 0.0);
    
    return weights;
  }
  
  private Dataset<String, String> initializeZDataset(int totalSentences, int[][] zLabels, int[][][] data) {
    int[][] flatData = new int[totalSentences][];
    int count = 0;
    for (int i = 0; i < data.length; i++) {
      for (int s = 0; s < data[i].length; s++)
        flatData[count++] = data[i][s];
    }
    
    int[] flatZLabels = flatten(zLabels, totalSentences);
    Log.severe("Created the Z dataset with " + flatZLabels.length + " datums.");
    
    return new Dataset<String, String>(zLabelIndex, flatZLabels, featureIndex, flatData);
  }

  private Map<String, RVFDataset<String, String>> initializeYDatasets() {
    Map<String, RVFDataset<String, String>> result = new HashMap<String, RVFDataset<String, String>>();
    for (String yLabel : yLabelIndex.objectsList())
      result.put(yLabel, new RVFDataset<String, String>());
    return result;
  }
  
  void predictZLabels(int [][] group,
      int[] zLabels,
      LinearClassifier<String, String> zClassifier) {
    for (int s = 0; s < group.length; s++) {
      Counter<String> probs = zClassifier.logProbabilityOf(group[s]);
      zLabels[s] = zLabelIndex.indexOf(Counters.argmax(probs));
    }
  }

  void computeZLogProbs(int[][] group, 
      Counter<String> [] zLogProbs, 
      LinearClassifier<String, String> zClassifier,
      int epoch) {
    for (int s = 0; s < group.length; s ++) {
      zLogProbs[s] = zClassifier.logProbabilityOf(group[s]);      
    }
  }
  
  /** updates the zLabels array with new predicted z labels */
  void inferZLabelsStable(int[][] group, 
      Set<Integer> positiveLabels,
      Set<Integer> negativeLabels, 
      int[] zLabels,
      Counter<String> [] zLogProbs, 
      LinearClassifier<String, String> zClassifier,
      int epoch) {
    boolean showProbs = false;
    boolean verbose = true;
    
    if(verbose) {
      System.err.print("inferZLabels: ");
      if(positiveLabels.size() > 1) System.err.println("MULTI RELATION");
      else if(positiveLabels.size() == 1) System.err.println("SINGLE RELATION");
      else System.err.println("NIL RELATION");
      System.err.println("positiveLabels: " + positiveLabels);
      System.err.println("negativeLabels: " + negativeLabels);
      System.err.print("Current zLabels:");
      for(int i = 0; i < zLabels.length; i ++) System.err.print(" " + zLabels[i]);
      System.err.println();
    }
    
    // compute the Z probabilities; these do not change
    computeZLogProbs(group, zLogProbs, zClassifier, epoch);
    
    for (int s = 0; s < group.length; s++) {
      double maxProb = Double.NEGATIVE_INFINITY;
      int bestLabel = -1;

      Counter<String> zProbabilities = zLogProbs[s]; 
      Counter<String> jointProbabilities = new ClassicCounter<String>();
      
      int origZLabel = zLabels[s];
      for (String candidate : zProbabilities.keySet()) {
        int candidateIndex = zLabelIndex.indexOf(candidate);
        
        // start with z probability
        if(showProbs) System.err.println("\tProbabilities for z[" + s + "]:");
        double prob = zProbabilities.getCount(candidate);
        zLabels[s] = candidateIndex;
        if(showProbs) System.err.println("\t\tlocal (" + zLabels[s] + ") = " + prob);
        
        // add the y probabilities
        for (int y : positiveLabels) {
          String yLabel = yLabelIndex.get(y);
          Datum<String, String> yDatum = 
            new RVFDatum<String, String>(extractYFeatures(yLabel, zLabels, zLogProbs), "");
          Counter<String> yProbabilities = yClassifiers.get(yLabel).logProbabilityOf(yDatum);
          double v = yProbabilities.getCount(yLabel);
          if(showProbs) System.err.println("\t\t\ty+ (" + y + ") = " + v);
          prob += v;
        }
        for (int y : negativeLabels) {
          String yLabel = yLabelIndex.get(y);
          Datum<String, String> yDatum = 
            new RVFDatum<String, String>(extractYFeatures(yLabel, zLabels, zLogProbs), "");
          Counter<String> yProbabilities = yClassifiers.get(yLabel).logProbabilityOf(yDatum);
          double v = yProbabilities.getCount(RelationMention.UNRELATED);
          if(showProbs) System.err.println("\t\t\ty- (" + y + ") = " + v);
          prob += v;
        }
        
        if(showProbs) System.err.println("\t\ttotal (" + zLabels[s] + ") = " + prob);
        jointProbabilities.setCount(candidate, prob);

        // update the current maximum
        if (prob > maxProb) {
          maxProb = prob;
          bestLabel = zLabels[s];
        }
      }

      if(bestLabel != -1 && bestLabel != origZLabel) {
        // found the best flip for this mention
        if(verbose) System.err.println("\tNEW zLabels[" + s + "] = " + bestLabel);
        zLabels[s] = bestLabel;
        zUpdatesInOneEpoch ++;
      } else {
        // nothing good found
        zLabels[s] = origZLabel;
      }
    } // end scan for group    
  }
  
  /** updates the zLabels array with new predicted z labels */
  void inferZLabels(int[][] group, 
      Set<Integer> positiveLabels,
      Set<Integer> negativeLabels, 
      int[] zLabels,
      Counter<String> [] zLogProbs, 
      LinearClassifier<String, String> zClassifier,
      int epoch) {
    boolean showProbs = false;
    boolean verbose = true;
    
    if(verbose) {
      System.err.print("inferZLabels: ");
      if(positiveLabels.size() > 1) System.err.println("MULTI RELATION");
      else if(positiveLabels.size() == 1) System.err.println("SINGLE RELATION");
      else System.err.println("NIL RELATION");
      System.err.println("positiveLabels: " + positiveLabels);
      System.err.println("negativeLabels: " + negativeLabels);
      System.err.print("Current zLabels:");
      for(int i = 0; i < zLabels.length; i ++) System.err.print(" " + zLabels[i]);
      System.err.println();
    }
    
    // compute the Z probabilities; these do not change
    computeZLogProbs(group, zLogProbs, zClassifier, epoch);

    // hill climbing until convergence
    // this is needed to guarantee labels that are consistent with "at least once"
    Set<Integer> flipped = new HashSet<Integer>();
    while(true){
      double maxProbGlobal = Double.NEGATIVE_INFINITY;
      int bestLabelGlobal = -1;
      int bestSentence = -1;
      
      for (int s = 0; s < group.length; s++) {
        if(flipped.contains(s)) continue;
        double maxProb = Double.NEGATIVE_INFINITY;
        int bestLabel = -1;

        Counter<String> zProbabilities = zLogProbs[s]; 
        Counter<String> jointProbabilities = new ClassicCounter<String>();

        int oldZLabel = zLabels[s];
        for (String candidate : zProbabilities.keySet()) {
          // start with z probability
          if(showProbs) System.err.println("\tProbabilities for z[" + s + "]:");
          double prob = zProbabilities.getCount(candidate);
          zLabels[s] = zLabelIndex.indexOf(candidate); 
          if(showProbs) System.err.println("\t\tlocal (" + zLabels[s] + ") = " + prob);
          
          // add the y probabilities
          for (int y : positiveLabels) {
            String yLabel = yLabelIndex.get(y);
            Datum<String, String> yDatum = 
              new RVFDatum<String, String>(extractYFeatures(yLabel, zLabels, zLogProbs), "");
            Counter<String> yProbabilities = yClassifiers.get(yLabel).logProbabilityOf(yDatum);
            double v = yProbabilities.getCount(yLabel);
            if(showProbs) System.err.println("\t\t\ty+ (" + y + ") = " + v);
            prob += v;
          }
          for (int y : negativeLabels) {
            String yLabel = yLabelIndex.get(y);
            Datum<String, String> yDatum = 
              new RVFDatum<String, String>(extractYFeatures(yLabel, zLabels, zLogProbs), "");
            Counter<String> yProbabilities = yClassifiers.get(yLabel).logProbabilityOf(yDatum);
            double v = yProbabilities.getCount(RelationMention.UNRELATED);
            if(showProbs) System.err.println("\t\t\ty- (" + y + ") = " + v);
            prob += v;
          }
          
          if(showProbs) System.err.println("\t\ttotal (" + zLabels[s] + ") = " + prob);
          jointProbabilities.setCount(candidate, prob);

          // update the current maximum
          if (prob > maxProb) {
            maxProb = prob;
            bestLabel = zLabels[s];
          }
        }
        //reset; we flip only the global best
        zLabels[s] = oldZLabel;

        // if we end up with a uniform distribution it means we did not predict anything. do not update
        if(bestLabel != -1 && bestLabel != zLabels[s] && 
           ! uniformDistribution(jointProbabilities) &&
           maxProb > maxProbGlobal) {
          // found the best flip so far
          maxProbGlobal = maxProb;
          bestLabelGlobal = bestLabel;
          bestSentence = s;
        } 
      } // end this group scan
      
      // no changes found; we converged
      if(bestLabelGlobal == -1) break;
      
      // flip one Z
      assert(bestSentence != -1);
      zLabels[bestSentence] = bestLabelGlobal;
      zUpdatesInOneEpoch ++;
      flipped.add(bestSentence);
      if(verbose) System.err.println("\tNEW zLabels[" + bestSentence + "] = " + zLabels[bestSentence]);
      
    } // end convergence loop    
    
    // check for flips that didn't happen
    boolean missedY = false;
    for(Integer y: positiveLabels) {
      boolean found = false;
      for(int i = 0; i < zLabels.length; i ++){
        if(zLabels[i] == y){
          found = true;
          break;
        }
      }
      if(! found) {
        missedY = true;
        break;
      }
    }
    if(verbose && missedY) {
      if(zLabels.length < positiveLabels.size()) {
        System.err.println("FOUND MISSED Y, smaller Z");
      } else {
        System.err.println("FOUND MISSED Y, larger Z");
      }
    }
  }
  
  private static boolean uniformDistribution(Counter<String> probs) {
    List<String> keys = new ArrayList<String>(probs.keySet());
    if(keys.size() < 2) return false;
    double p = probs.getCount(keys.get(0));
    for(int i = 1; i < keys.size(); i ++){
      if(p != probs.getCount(keys.get(i))){
        return false;
      }
    }
    return true;
  }
  
  Counter<String> extractYFeatures(
      String yLabel, 
      int[] zLabels, 
      Counter<String> [] zLogProbs) {
    if(featureModel == 0) {
      // this corresponds to MIML-RE AtLeastOnce
      return extractYFeaturesBoolean(yLabel, zLabels, zLogProbs, false);
    }

    if(featureModel == 1) {
      // this is MIML-RE (with label dependencies)
      return extractYFeaturesBoolean(yLabel, zLabels, zLogProbs, true);
    }

    throw new RuntimeException("ERROR: unknown feature model " + featureModel);
  }

  private Counter<String> extractYFeaturesBoolean(
      String yLabel,
      int[] zLabels,
      Counter<String> [] zLogProbs,
      boolean addDependencies) {
    assert(! yLabel.equals(RelationMention.UNRELATED));
    int count = 0;
    Set<String> others = new HashSet<String>();
    for (int s = 0; s < zLabels.length; s ++) {
      String zString = zLabelIndex.get(zLabels[s]);
      if (zString.equals(yLabel)) {
        count ++;
      } else if(! zString.equals(RelationMention.UNRELATED)) {
        others.add(zString);
      }
    }

    Counter<String> features = new ClassicCounter<String>();

    // was this label proposed by at least a Z?
    if (count > 0) {
      features.setCount(ATLEASTONCE_FEAT, 1.0);
    }
    // no Z proposed this label
    else if(count == 0){
      features.setCount(NONE_FEAT, 1.0);
    }

    // label dependencies
    if(addDependencies && count > 0) {
      for(String z: others) {
        String f = makeCoocurrenceFeature(yLabel, z);
        if(true) {
          features.setCount(f, 1.0);
        }
      }
    }

    return features;
  }

  public static List<Pair<String, Double>> sortPredictions(Counter<String> scores) {
    List<Pair<String, Double>> sortedScores = new ArrayList<Pair<String,Double>>();
    for(String key: scores.keySet()) {
      sortedScores.add(new Pair<String, Double>(key, scores.getCount(key)));
    }
    sortPredictions(sortedScores);
    return sortedScores;
  }
  
  private static void sortPredictions(List<Pair<String, Double>> scores) {
    Collections.sort(scores, new Comparator<Pair<String, Double>>() {
      @Override
      public int compare(Pair<String, Double> o1, Pair<String, Double> o2) {
        if(o1.second() > o2.second()) return -1;
        if(o1.second().equals(o2.second())){
          // this is an arbitrary decision to disambiguate ties
          int c = o1.first().compareTo(o2.first()); 
          if(c < 0) return -1;
          else if(c == 0) return 0;
          return 1;
        }
        return 1;
      }
    });
  }
  
  /**
   * Implements weighted voting over the different Z classifiers in each fold
   * @return Probabilities (NOT log probs!) for each known label
   */
  private Counter<String> classifyLocally(Collection<String> sentence) {
    Datum<String, String> datum = new BasicDatum<String, String>(sentence);

    if(localClassificationMode == LOCAL_CLASSIFICATION_MODE.WEIGHTED_VOTE) {
      Counter<String> sumProbs = new ClassicCounter<String>();
      
      for(int fold = 0; fold < numberOfFolds; fold ++) {
        LinearClassifier<String, String> zClassifier = zClassifiers[fold];
        Counter<String> probs = zClassifier.probabilityOf(datum);
        sumProbs.addAll(probs);
      }
      
      for(String l: sumProbs.keySet()) 
        sumProbs.setCount(l, sumProbs.getCount(l) / numberOfFolds);
      return sumProbs;
    }
    
    if(localClassificationMode == LOCAL_CLASSIFICATION_MODE.SINGLE_MODEL) {
      Counter<String> probs = zSingleClassifier.probabilityOf(datum);
      return probs;
    }
    
    throw new RuntimeException("ERROR: classification mode " + localClassificationMode + " not supported!");
  }
  
  @Override
  public Counter<String> classifyOracleMentions(
      List<Collection<String>> sentences,
      Set<String> goldLabels) {
    Counter<String> [] zProbs = 
      ErasureUtils.uncheckedCast(new Counter[sentences.size()]);
      
    //
    // Z level predictions
    //
    Counter<String> yLabels = new ClassicCounter<String>();
    Map<String, Counter<Integer>> ranks = new HashMap<String, Counter<Integer>>();
    for (int i = 0; i < sentences.size(); i++) {
      Collection<String> sentence = sentences.get(i);
      zProbs[i] = classifyLocally(sentence);
      
      if(! uniformDistribution(zProbs[i])) {
        List<Pair<String, Double>> sortedProbs = sortPredictions(zProbs[i]);
        double top = sortedProbs.get(0).second();
        for(int j = 0; j < sortedProbs.size() && j < 3; j ++) {
          String l = sortedProbs.get(j).first();
          double v = sortedProbs.get(j).second();
          if(v + 0.99 < top) break;
          if(! l.equals(RelationMention.UNRELATED)){ // && v + 0.99 > top){// && goldLabels.contains(l)) {
            double rank = 1.0 / (1.0 + (double) j);
            Counter<Integer> lRanks = ranks.get(l);
            if(lRanks == null) {
              lRanks = new ClassicCounter<Integer>();
              ranks.put(l, lRanks);
            }
            lRanks.setCount(j, rank);
          }
        }
      }
    }
    
    for(String l: ranks.keySet()) {
      double sum = 0;
      for(int position: ranks.get(l).keySet()) {
        sum += ranks.get(l).getCount(position);
      }
      double rank = sum / sentences.size(); // ranks.get(l).keySet().size();
      System.err.println("RANK = " + rank);
      if(rank >= 0) // 0.001)
        yLabels.setCount(l, rank);
    }
    
    return yLabels;
  }
  
  @Override
  public Counter<String> classifyMentions(List<Collection<String>> sentences) {
    String[] zLabels = new String[sentences.size()];
    Counter<String> [] zLogProbs = 
      ErasureUtils.uncheckedCast(new Counter[sentences.size()]);
      
    //
    // Z level predictions
    //
    Counter<String> localSum = new ClassicCounter<String>();
    Counter<String> localBest = new ClassicCounter<String>();
    Counter<String> localNoisyOr = new ClassicCounter<String>();
    for (int i = 0; i < sentences.size(); i++) {
      Collection<String> sentence = sentences.get(i);
      Counter<String> probs = classifyLocally(sentence);
      
      zLogProbs[i] = new ClassicCounter<String>();
      for(String l: probs.keySet()) {
        zLogProbs[i].setCount(l, Math.log(probs.getCount(l)));
      }
      
      List<Pair<String, Double>> sortedProbs = sortPredictions(probs);
      Pair<String, Double> prediction = sortedProbs.get(0);
      String l = prediction.first();
      double s = prediction.second();
      zLabels[i] = l;
      // we do not output NIL labels
      if(! zLabels[i].equals(RelationMention.UNRELATED)) {
        localSum.incrementCount(l, s);
        if(! localBest.containsKey(l) || s > localBest.getCount(l)) {
          localBest.setCount(l, s);
        }
        
        double crt = (localNoisyOr.containsKey(l) ? localNoisyOr.getCount(l) : 1.0);
        crt = crt * (1.0 - s);
        localNoisyOr.setCount(l, crt);
      }
      // System.err.println("***zLabels[" + i + "]:" + zLabels[i]);
    }
    
    int[] zLabelIndices = new int[zLabels.length];
    for (int i = 0; i < zLabels.length; i++)
      zLabelIndices[i] = zLabelIndex.indexOf(zLabels[i]);
    
    //
    // Y level predictions
    //
    Counter<String> result = new ClassicCounter<String>();
    for (String yLabel : yClassifiers.keySet()) {
      LinearClassifier<String, String> yClassifier = yClassifiers.get(yLabel);
      Counter<String> features = extractYFeatures(yLabel, zLabelIndices, zLogProbs);
      Datum<String, String> datum = new RVFDatum<String, String>(features, "");
      Counter<String> probs = yClassifier.probabilityOf(datum);
      double posScore = probs.getCount(yLabel);
      double negScore = probs.getCount(RelationMention.UNRELATED);       
      //System.err.println("***POS SCORE: " + posScore + "***");
      //System.err.println("***NEG SCORE: " + negScore + "***");
      if (posScore > negScore)
        result.incrementCount(yLabel, posScore);
    }
    
    // this score is not a proper probability,
    // but it is very useful to disambiguate ties
    Counter<String> joint = new ClassicCounter<String>();
    for(String l: result.keySet()) {
      // New ranking score using NoisyOr:
      double yProb = 1.0; // result.getCount(l);
      double zProb = (1.0 - localNoisyOr.getCount(l)); 
      joint.setCount(l, yProb * zProb);
      
      // Old ranking score:
      //double trueYProb = result.getCount(l);
      //double zProbSum = localSum.getCount(l);
      //joint.setCount(l, trueYProb * zProbSum);
    }
    
    // return result;
    return joint;
  }

  public void save(String path) throws IOException {
    // make sure the directory specified by path exists
    int lastSlash = path.lastIndexOf(File.separator);
    if (lastSlash > 0) {
      File dir = new File(path.substring(0, lastSlash));
      if (! dir.exists())
        dir.mkdirs();
    }
    
    ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path));
    out.writeObject(knownDependencies);
    out.writeObject(zLabelIndex);
    out.writeInt(zClassifiers.length);
    for(int i = 0; i < zClassifiers.length; i ++)
      out.writeObject(zClassifiers[i]);
    out.writeObject(zSingleClassifier);
    out.writeInt(yClassifiers.keySet().size());
    for (String yLabel : yClassifiers.keySet()) {
      out.writeObject(yLabel);
      out.writeObject(yClassifiers.get(yLabel));
    }
    out.close();
  }
  
  @Override
  public void load(ObjectInputStream in) throws IOException, ClassNotFoundException {
    knownDependencies = ErasureUtils.uncheckedCast(in.readObject());
    zLabelIndex = ErasureUtils.uncheckedCast(in.readObject());
    
    numberOfFolds = in.readInt();
    zClassifiers = ErasureUtils.uncheckedCast(new LinearClassifier[numberOfFolds]);
    for(int i = 0; i < numberOfFolds; i ++){
      LinearClassifier<String, String> classifier =
        ErasureUtils.uncheckedCast(in.readObject());
      zClassifiers[i] = classifier;
    }
    zSingleClassifier = 
      ErasureUtils.uncheckedCast(in.readObject());

    int numLabels = in.readInt();
    yClassifiers = new HashMap<String, LinearClassifier<String, String>>();
    for (int i = 0; i < numLabels; i++) {
      String yLabel = ErasureUtils.uncheckedCast(in.readObject());
      LinearClassifier<String, String> classifier = 
        ErasureUtils.uncheckedCast(in.readObject());
      yClassifiers.put(yLabel, classifier);
      Log.severe("Loaded Y classifier for label " + yLabel + 
          ": " + classifier.toAllWeightsString());
    }
  }
  
  public static JointlyTrainedRelationExtractor load(String path, Properties props) throws IOException, ClassNotFoundException {
    ObjectInputStream in = new ObjectInputStream(new FileInputStream(path));
    JointBayesRelationExtractor extractor = new JointBayesRelationExtractor(props); 
    extractor.load(in);
    in.close();
    return extractor;
  }  
}