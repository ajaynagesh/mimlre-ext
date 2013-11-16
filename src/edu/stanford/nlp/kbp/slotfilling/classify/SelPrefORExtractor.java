package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.kbp.slotfilling.KBPTrainer;
import edu.stanford.nlp.kbp.slotfilling.classify.HoffmannExtractor.LabelWeights;
import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.sequences.SequenceGibbsSampler;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.StringUtils;

import java.io.*;
import java.util.*;

/**
 * Our version of the distant supervision relation extraction system.
 */
/**
 * Implements as closely as possible the MultiR algorithm from (Hoffmann et al., 2011)
 * @author Mihai
 */
public class SelPrefORExtractor extends JointlyTrainedRelationExtractor {
  private static final long serialVersionUID = 1L;
  private static final int LABEL_ALL = -1;

  private static int ALGO_TYPE = 1; // 1 - for test type 1 , 2 - for test type 2 (cf. write-up,notes on google doc)
  
  /**
   * Stores weight information for one label
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
    public void addToAverage() {
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
  
  /** Stores weight information for the entity types participating in the relation */
  LabelWeights [] tWeights_arg1;
  LabelWeights [] tWeights_arg2;

  Index<String> labelIndex;
  Index<String> zFeatureIndex;
  /** Index of the NIL label */
  int nilIndex;
  /** Number of epochs during training */
  final int epochs;
  
  int epochsInf;

  public SelPrefORExtractor(int epochs) {
    this.epochs = epochs;
  }
  public SelPrefORExtractor(Properties props) {
    Log.severe("SelPrefORExtractor configured with the following properties:");
    this.epochs = PropertiesUtils.getInt(props, Props.PERCEPTRON_EPOCHS, 10); // TODO: To add relevant information to our training algorithm
    Log.severe("epochs = " + epochs);
  }

  @Override
  public void train(MultiLabelDataset<String, String> dataset) {
  	
	  	// 4 sets of variables for each of the factors
	  	// 1) f_bias 2) f_select 3) f_mention 4) f_extract
	  
	  	/*
	  	 * f_bias -- independent
	  	 * f_select (T_arg1, T_arg2, Y)
	  	 * f_mention (Y, Z_1 .. Z_n)
	  	 * f_extract -- independent -- DONE
	  	 */
	  
	    Log.severe("Training the \"Selectional Preference with Overlapping relations\" model using "
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

	    // Weights for the entity types in a relation (one for each type of entity)
	    tWeights_arg1 = new LabelWeights[dataset.argTypeIndex().size()];
	    for(int i = 0; i < tWeights_arg1.length; i++) 
	    	tWeights_arg1[i] = new LabelWeights(dataset.argFeatIndex().size());
	    
	    // repeat for a number of epochs
	    for(int t = 0; t < epochs; t ++){
	    	// randomize the data set in each epoch
	    	// use a fixed seed for replicability
	    	Log.severe("Started epoch #" + t + "...");
	    	dataset.randomize(t);

	    	// TODO: Check -- Need to update some statistics ??
	    	Counter<Integer> posUpdateStats = new ClassicCounter<Integer>();
	    	Counter<Integer> negUpdateStats = new ClassicCounter<Integer>();

	    	// traverse the relation dataset
	    	for(int i = 0; i < dataset.size(); i ++){
	    		int [][] crtGroup = dataset.getDataArray()[i];
	    		Set<Integer> goldPos = dataset.getPositiveLabelsArray()[i];

	    		Set<Integer> arg1Type = dataset.arg1TypeArray()[i];
	    		Set<Integer> arg2Type = dataset.arg2TypeArray()[i];

	    		trainJointly(crtGroup, goldPos, arg1Type, arg2Type, posUpdateStats, negUpdateStats);

	    		// update the number of iterations an weight vector has survived
	    		for(LabelWeights zw: zWeights) zw.updateSurvivalIterations();
	    	}

	    	Log.severe("Epoch #" + t + " completed. Inspected " +
	    			dataset.size() + " datum groups. Performed " +
	    			posUpdateStats.getCount(LABEL_ALL) + " ++ updates and " +
	    			negUpdateStats.getCount(LABEL_ALL) + " -- updates.");
	    	
	    }

	    // finalize learning: add the last vector to the avg for each label
	    for(LabelWeights zw: zWeights) zw.addToAverage();
  }
  
  static public List<File> fetchFiles(String path, final String extension) {
	  File kbDir = new File(path);
	  assert (kbDir.isDirectory());
	  File[] inputFiles = kbDir.listFiles(new FileFilter() {
		  @Override
		  public boolean accept(File pathname) {
			  if (pathname.getAbsolutePath().endsWith(extension))
				  return true;
			  return false;
		  }
	  });
	  List<File> files = Arrays.asList(inputFiles);
	  Collections.sort(files, new Comparator<File>() {
		  @Override
		  public int compare(File o1, File o2) {
			  return o1.getAbsolutePath().compareTo(o2.getAbsolutePath());
		  }
	  });
	  return files;
  }
  
  public static void train(Properties props) throws Exception {
	  File workDir = new File(props.getProperty(Props.WORK_DIR));
	  assert (workDir.isDirectory());
	  File trainDir = new File(workDir + File.separator + "train");
	  RelationExtractorFactory factory = 
			  new RelationExtractorFactory(props.getProperty(Props.MODEL_TYPE, Constants.DEFAULT_MODEL));
	  Log.severe("modelType = " + factory.modelType());
	  double samplingRatio = PropertiesUtils.getDouble(props, 
			  Props.NEGATIVES_SAMPLE_RATIO,
			  Constants.DEFAULT_NEGATIVES_SAMPLING_RATIO);

	  KBPTrainer trainer = new KBPTrainer(props, true);
	  if (!trainer.modelExists()) {
		  // construct dataset and train
		  List<File> trainDatumFiles = fetchFiles(trainDir.getAbsolutePath(), ".datums");

		  // generate a separate file with negative examples, so we have a fixed dataset for training
		  File negFile = null;
		  if(Constants.OFFLINE_NEGATIVES) {
			  negFile = new File(trainDir + File.separator + 
					  "datums_" + (int) (100.0 * samplingRatio) + ".negatives");
			  if(! negFile.exists()) {
				  KBPTrainer.subsampleNegatives(trainDatumFiles, negFile, samplingRatio);
			  }
		  }

		  if (factory.isLocallyTrained()) {
			  trainer.trainOneVsAll(trainDatumFiles, negFile);
		  } else {
			  assert(Constants.OFFLINE_NEGATIVES);
			  trainer.trainAtLeastOnce(props, trainDatumFiles, negFile);
		  } 
		  Log.severe("Training complete.");
	  }
  }
  
  public static void main(String[] args) throws Exception{

	  Properties props = StringUtils.argsToProperties(args);
	  Log.setLevel(Log.stringToLevel(props.getProperty(Props.LOG_LEVEL)));
	  Log.severe("--------------Running the new algo ---- One small step for man .... :-) ");
	  Log.severe("Using run id: " + props.getProperty(Props.RUN_ID) + " in working directory " + props.getProperty(Props.WORK_DIR));

	  train(props);

  }
  
  /*
   * while (not_converged){								    
							Step 1: // update Z vars assuming Y vars to be fixed .. 			   
										done by simple enumeration   			   
									   
							Step 2: // update Y assuming Z to be fixed .. done by Gibbs Sampling	   
								while (not_converged){						   
									choose a random permutation of Y variables → \Pi
									for ( i = 1 to |Y| ) {						   
											update Y_{\Pi_i} assuming the rest is fixed		    
									}								    
								}									    
						}
   * 
   */
  

  private Counter<Integer> estimateZ(int [] datum) {
    Counter<Integer> vector = new ClassicCounter<Integer>();
    for(int d: datum) vector.incrementCount(d);

    Counter<Integer> scores = new ClassicCounter<Integer>();
    for(int label = 0; label < zWeights.length; label ++){
      double score = zWeights[label].dotProduct(vector);
      scores.setCount(label, score);
    }

    return scores;
  }
  /**
   * New estimateZ function which computes the factors given Yi
   * @param datum
   * @param Yi
   * @return
   */
  private Counter<Integer> estimateZ(int [] datum, Set<Integer> Yi) {
	    Counter<Integer> vector = new ClassicCounter<Integer>();
	    for(int d: datum) vector.incrementCount(d);
	    
	    Counter<Integer> Yvector = new ClassicCounter<Integer>();
	    for(int y : Yi) vector.incrementCount(y);

	    Counter<Integer> scores = new ClassicCounter<Integer>();
	    
	    for(int label = 0; label < zWeights.length; label ++){
	      double score = zWeights[label].dotProduct(vector);
	      score += zWeights[label].dotProduct(Yvector); // note the addition to the score
	      scores.setCount(label, score);
	    }

	    return scores;
	  }
  
  private List<Counter<Integer>> ComputePrZ_Yi (int [][] datums, Set<Integer> Yi) {
	  List<Counter<Integer>> zs = new ArrayList<Counter<Integer>>();
	  
	  for(int [] datum: datums) {
	      zs.add(estimateZ(datum, Yi));
	  }
	  
	  return zs;
  }
  
//\hat{Y,Z} = argmax_{Y,Z} Pr_{\theta} (Y, Z | T_i, x_i)
  private void ComputePrYZ_Ti (int [][] datums, int szY, Set<Integer> arg1Type, Set<Integer> arg2Type) {
	  Set<Integer> yPredicted = null; //  TODO: Initialize yPredicted. How ? 
	  
	  for (int i = 0; i < epochsInf; i ++){ // TODO: What is the stopping criterion
		  
		  // update Z assuming Y fixed 
		  List<Counter<Integer>> zs = ComputePrZ_Yi(datums, yPredicted);
		  int [] zPredicted = generateZPredicted(zs);
		  
		  // update Y assuming Z fixed (to zPredicted)
		  for (int j = 0; j < epochsInf; j++){ // TODO: What is the stopping criterion
			  Counter<Integer> randomizedY = randomizeVar();
			  
			  for(int k = 0; k < szY; k ++){
				  Counter<Integer> y = estimateY(zPredicted, szY, arg1Type, arg2Type);
				  
			  }
		  }
		  
	  }
  }
  
  private void ComputePrZT_Yi () {
	  
  }
  
  private void ComputePrYZT (List<Counter<Integer>> pr_y, List<Counter<Integer>> pr_z, List<Counter<Integer>> pr_t) {
	  
  }
  
  private void computeFactor() {
	  
  }
  
  // TODO: Need to implement the randomization routine
  private Counter<Integer> randomizeVar(){
	  Counter<Integer> rVar = new ClassicCounter<Integer>();
	  
	  return rVar;
  }
  
  /*private int [] gibbsSampler(int [] zPredicted){
	  
	  // TODO: check - init. to be done here or in constructor ??
	  LabelWeights [] yVarsWts = new LabelWeights[10]; //TODO: No. of Y vars need to be intialised
	  for(LabelWeights yWt : yVarsWts){
		  yWt = new LabelWeights(labelIndex.size());
	  }
	  
	  
	  int [] yPredicted = new int[10]; // TODO: Init the correct val
	  
	  
	  for(int i = 0; i < epochsInf; i++){ // TODO: Need to determine the right val. or another stopping criterion
		  Counter<Integer> randomizedY = randomizeVar();
		  
		  for(int yVal : randomizedY.keySet()){
			  computeFactor();
		  }
		  
	  }
	  
	  return yPredicted;
  }*/
  
  List<Counter<Integer>> ComputePrY_ZiTi(int [] zPredicted, Set<Integer> arg1Type, Set<Integer> arg2Type){
	  List<Counter<Integer>> ys = new ArrayList<Counter<Integer>>();
	  
	  return ys;
  }
  
  void generateYZPredicted(List<Counter<Integer>> ys, List<Counter<Integer>> zs, Counter<Integer> yPredicted, int [] zPredicted) {
	  yPredicted = new ClassicCounter<Integer>();
	  
  }
  
  void generateYZTPredicted(List<Counter<Integer>> ys, List<Counter<Integer>> zs, List<Counter<Integer>> ts, 
		  Counter<Integer> yPredicted, int [] zPredicted, int [] tPredicted) {
	  
  }
  List<Counter<Integer>> ComputePrZ(int [][] crtGroup) {
	  List<Counter<Integer>> prZs = estimateZ(crtGroup);
	  
	  return prZs;
  }
  
  private void trainJointly(
		  int [][] crtGroup,
          Set<Integer> goldPos,
          Set<Integer> arg1Type,
          Set<Integer> arg2Type,
          Counter<Integer> posUpdateStats,
          Counter<Integer> negUpdateStats) {
	  
	  int [] zPredicted = null;
	  int [] tPredicted = null;
	  
	  if(ALGO_TYPE == 1){
		  // \hat{Y,Z} = argmax_{Y,Z} Pr_{\theta} (Y, Z | T_i, x_i)
		  // 1. estimate Pr(Z) .. for now estimating \hat{Z}
		  List<Counter<Integer>> pr_z = ComputePrZ(crtGroup);
		  zPredicted = generateZPredicted(pr_z);
		  // 2. estimate Pr(Y|Z,T)
		  List<Counter<Integer>> pr_y = ComputePrY_ZiTi(zPredicted, arg1Type, arg2Type);
		  
		  Counter<Integer> yPredicted = null;
		  
		  generateYZPredicted(pr_y, pr_z, yPredicted, zPredicted);
		  
		  Set<Integer> [] zUpdate;
		  
		  if(updateCondition(yPredicted.keySet(), goldPos)){
			  
			  zUpdate = generateZUpdate(goldPos, pr_z);
			  updateZModel(zUpdate, zPredicted, crtGroup, posUpdateStats, negUpdateStats);
		  }  
	  }
	  
	  else if(ALGO_TYPE == 2){
		  List<Counter<Integer>> pr_y = null;
		  List<Counter<Integer>> pr_z = null;
		  List<Counter<Integer>> pr_t = null;
		  
		  Counter<Integer> yPredicted = null;
		  
		  Set<Integer> [] zUpdate;
		  Set<Integer> [] tUpdate;
		  
		  ComputePrYZT(pr_y, pr_z, pr_t);
		  generateYZTPredicted(pr_y, pr_z, pr_t, yPredicted, zPredicted, tPredicted);
		  
		  if(updateCondition(yPredicted.keySet(), goldPos)){
			  
			  zUpdate = generateZUpdate(goldPos, pr_z);
			  tUpdate = generateTUpdate(goldPos, pr_z);
			  updateZModel(zUpdate, zPredicted, crtGroup, posUpdateStats, negUpdateStats);
		  } 
		  
	  }
	  
  }
  

  Set<Integer> [] generateTUpdate(Set<Integer> goldPos,List<Counter<Integer>> zs)
  {
	  Set<Integer> [] tUpdate = null;
	  
	  return tUpdate;
  }
  
  int[] generateTPredicted(){
	  
	  int []tPredicted = null;
	  
	  return tPredicted;
  }
  
  private void updateZModel(
          Set<Integer> [] goldZ,
          int [] predictedZ,
          int [][] group,
          Counter<Integer> posUpdateStats,
          Counter<Integer> negUpdateStats) {
    assert(goldZ.length == group.length);
    assert(predictedZ.length == group.length);

    for(int i = 0; i < group.length; i ++) {
      // list of all possible gold labels for this mention (z)
      // for theoretical reasons this is a set, but in practice it will have a single value
      // also, for NIL labels, this set is empty
      Set<Integer> gold = goldZ[i];
      int pred = predictedZ[i];
      int [] datum = group[i];

      // negative update
      if(pred != nilIndex && ! gold.contains(pred)) {
        zWeights[pred].update(datum, -1.0);
        negUpdateStats.incrementCount(pred);
        negUpdateStats.incrementCount(LABEL_ALL);
      }

      // negative update for NIL
      if(pred == nilIndex && gold.size() != 0){
        zWeights[nilIndex].update(datum, -1.0);
        negUpdateStats.incrementCount(pred);
      }

      // positive update(s)
      for(int l: gold) {
        if(l != nilIndex && l != pred) {
          zWeights[l].update(datum, +1.0);
          posUpdateStats.incrementCount(l);
          posUpdateStats.incrementCount(LABEL_ALL);
        }
      }

      // positive update for NIL
      if(gold.size() == 0 && pred != nilIndex){
        zWeights[nilIndex].update(datum, +1.0);
        posUpdateStats.incrementCount(nilIndex);
      }
    }
  }

  private static boolean updateCondition(Set<Integer> y, Set<Integer> yPos) {
    if(y.size() != yPos.size()) return true;

    for(Integer l: yPos) {
      if(! y.contains(l)) {
        return true;
      }
    }

    return false;
  }

  /*
   * Gibbs sampling of Z
   * TODO: This needs to be replaced by our inference code
   */
  private Set<Integer> [] generateZUpdate(
          Set<Integer> goldPos,
          List<Counter<Integer>> zs) {
    Set<Integer> [] zUpdate = ErasureUtils.uncheckedCast(new Set[zs.size()]);


    return zUpdate;
  }

  /**
   * TODO: Need to code this up correctly
   * 
   */
  private Counter<Integer> estimateY(int [] zPredicted, int ySz, Set<Integer> arg1Type, Set<Integer> arg2Type) {
    Counter<Integer> ys = new ClassicCounter<Integer>();
    
    LabelWeights yWeights = new LabelWeights(ySz);
    
    // arg1vector
    Counter<Integer> arg1TypeVector = new ClassicCounter<Integer>();
    for(int arg1 : arg1Type) arg1TypeVector.incrementCount(arg1);
    
    // arg2vector
    Counter<Integer> arg2TypeVector = new ClassicCounter<Integer>();
    for(int arg2 : arg2Type) arg2TypeVector.incrementCount(arg2);
    
    // zpredVector
    Counter<Integer> zPredVector = new ClassicCounter<Integer>();
    for(int z : zPredicted) zPredVector.incrementCount(z);
    
    return ys;
  }

  private List<Counter<Integer>> estimateZ(int [][] datums) {
    List<Counter<Integer>> zs = new ArrayList<Counter<Integer>>();
    for(int [] datum: datums) {
      zs.add(estimateZ(datum));
    }
    return zs;
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

  private static int pickBestLabel(Counter<Integer> scores) {
    assert(scores.size() > 0);
    List<Pair<Integer, Double>> sortedScores = sortPredictions(scores);
    return sortedScores.iterator().next().first();
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
        if(o1.second() < o2.second()) return 1;

        // this is an arbitrary decision to disambiguate ties
        if(o1.first() > o2.first()) return -1;
        if(o1.first() < o2.first()) return 1;
        return 0;
      }
    });
  }

  @Override
  public Counter<String> classifyMentions(List<Collection<String>> mentions) {
    Counter<String> bestZScores = new ClassicCounter<String>();

    // traverse of all mention of this tuple
    for (int i = 0; i < mentions.size(); i++) {
      // get all scores for this mention
      Collection<String> mentionFeatures = mentions.get(i);
      Counter<String> mentionScores = classifyMention(mentionFeatures);

      Pair<String, Double> topPrediction = JointBayesRelationExtractor.sortPredictions(mentionScores).get(0);
      String l = topPrediction.first();
      double s = topPrediction.second();

      // update the best score for this label if necessary
      // exclude the NIL label from this; it is not propagated in the Y layer
      if(! l.equals(RelationMention.UNRELATED) &&
         (! bestZScores.containsKey(l) || bestZScores.getCount(l) < s)) {
        bestZScores.setCount(l, s);
      }
    }

    // generate the predictions of the Y layer using deterministic OR
    // the score of each Y label is the best mention-level score
    return bestZScores;
  }

  private Counter<String> classifyMention(Collection<String> testDatum) {
    Counter<String> scores = new ClassicCounter<String>();
    for(int labelIdx = 0; labelIdx < zWeights.length; labelIdx ++){
      double score = zWeights[labelIdx].avgDotProduct(testDatum, zFeatureIndex);
      scores.setCount(labelIndex.get(labelIdx), score);
    }
    return scores;
  }

  // TODO: Check : Do we need to save more information on the serial file of our training model
  @Override
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

  @Override
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
    SelPrefORExtractor ex = new SelPrefORExtractor(props);
    ex.load(in);
    in.close();
    is.close();
    return ex;
  }

}

