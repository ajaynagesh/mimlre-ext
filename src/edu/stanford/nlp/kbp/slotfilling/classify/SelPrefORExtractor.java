package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.kbp.slotfilling.KBPTrainer;
import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
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

  //TODO: need to add this in the param file
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
  LabelWeights [] arg1biasFweights;
  LabelWeights [] arg2biasFweights;

  LabelWeights selectFweights;  // Shared selectFweights vector for all the select factors
  LabelWeights mentionFweights; // Shared mentionFweights vector for all the mention factors
  
  //double eta;
  double getEta(int epoch){
	  return (1.0/epoch);
  }
  
  Index<String> labelIndex;
  Index<String> zFeatureIndex;
  Index<String> argTypeIndex;
  /** Index of the NIL label */
  int nilIndex;
  /** Number of epochs during training */
  final int epochs;
  
  int epochsInf = 10;

  public SelPrefORExtractor(int epochs) {
    this.epochs = epochs;
  }
  public SelPrefORExtractor(Properties props) {
    Log.severe("SelPrefORExtractor configured with the following properties:");
    this.epochs = PropertiesUtils.getInt(props, Props.PERCEPTRON_EPOCHS, 10);
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
	    argTypeIndex = dataset.argTypeIndex();

	    int numOfLabels = labelIndex.size();
	    
	    zWeights = new LabelWeights[numOfLabels];
	    for(int i = 0; i < zWeights.length; i ++)
	      zWeights[i] = new LabelWeights(dataset.featureIndex().size());

	    int numOfTypes = dataset.argTypeIndex.size();
	    selectFweights = new LabelWeights(numOfTypes*numOfTypes*numOfLabels+(2*numOfTypes*numOfLabels));
	    
	    mentionFweights = new LabelWeights (numOfLabels*numOfLabels);

	    // Weights for the entity types in a relation (one for each type of entity)
//	    arg1biasFweights = new LabelWeights[dataset.argTypeIndex().size()];
//	    for(int i = 0; i < arg1biasFweights.length; i++) 
//	    	arg1biasFweights[i] = new LabelWeights(dataset.argFeatIndex().size());
//	    
//	    arg2biasFweights = new LabelWeights[dataset.argTypeIndex().size()];
//	    for(int i = 0; i < arg2biasFweights.length; i++) 
//	    	arg2biasFweights[i] = new LabelWeights(dataset.argFeatIndex().size());

	    /**
	     * Training algorithm starts here
	     */
	    // repeat for a number of epochs
	    for(int t = 1; t <= epochs; t ++){
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

	    		trainJointly(crtGroup, goldPos, arg1Type, arg2Type, posUpdateStats, negUpdateStats, i, t);

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
  
  static void testRandomArrayGen(){
	  int abc[] = {1,2,3,4,5,6,7,8,9};
	  ArrayList<Integer> abcList = new ArrayList<Integer>();
	  for(int a : abc){
		  abcList.add(a);
	  }
	  System.out.println(abcList);
	  Collections.shuffle(abcList);
	  System.out.println(abcList);
	  Collections.shuffle(abcList);
	  System.out.println(abcList);
	  Collections.shuffle(abcList);
	  System.out.println(abcList);
	  Collections.shuffle(abcList);
	  System.out.println(abcList);
	  Collections.shuffle(abcList);
	  System.out.println(abcList);
	  Collections.shuffle(abcList);
	  System.out.println(abcList);
	  Collections.shuffle(abcList);
	  System.out.println(abcList);
	  Collections.shuffle(abcList);
	  System.out.println(abcList);
	  Collections.shuffle(abcList);
	  System.out.println(abcList);
	  Collections.shuffle(abcList);
	  System.out.println(abcList);
	  Collections.shuffle(abcList);
	  System.out.println(abcList);
	  Collections.shuffle(abcList);
	  System.out.println(abcList);
	  Collections.shuffle(abcList);
	  System.out.println(abcList);
	  
	  SelPrefORExtractor s = new SelPrefORExtractor(0);
	  double eta = -s.getEta(10);
	  System.out.println(eta);
	  System.out.println(+s.getEta(10));
	  
	  System.out.println(Math.exp(1023));
	  System.out.println(Double.MAX_EXPONENT);
	  
	  System.exit(0);
  }
  
  public static void main(String[] args) throws Exception{

	  Properties props = StringUtils.argsToProperties(args);
	  Log.setLevel(Log.stringToLevel(props.getProperty(Props.LOG_LEVEL)));
	  Log.severe("--------------Running the new algo ---- One small step for man .... :-) ");
	  Log.severe("Using run id: " + props.getProperty(Props.RUN_ID) + " in working directory " + props.getProperty(Props.WORK_DIR));

	  //testRandomArrayGen();
	  
	  train(props);

  }  

  private Counter<Integer> estimateZ(int [] datum) {
    Counter<Integer> vector = new ClassicCounter<Integer>();
    for(int d: datum) vector.incrementCount(d);

    Counter<Integer> scores = new ClassicCounter<Integer>();
    for(int label = 0; label < zWeights.length; label ++){
      double score = zWeights[label].dotProduct(vector);
      scores.setCount(label, score);
//      if (score > 0)
//    	  System.out.println("====================================We have a non-zero score");
    }

    return scores;
  }
  
  private double estimateZ(int [] datum, int label) {
	    Counter<Integer> vector = new ClassicCounter<Integer>();
	    for(int d: datum) 
	    	vector.incrementCount(d);
	    
	    double score = zWeights[label].dotProduct(vector);
	      
	    return score;
	  }
  
  private void ComputePrYZT (List<Counter<Integer>> pr_y, List<Counter<Integer>> pr_z, List<Counter<Integer>> pr_t) {
	  
  }
  
  private ArrayList<Integer> randomizeVar(int sz){
	  
	  ArrayList<Integer> randomArray = new ArrayList<Integer>();
	  for(int i = 0; i < sz; i ++)
		  randomArray.add(i);
	  Collections.shuffle(randomArray);

	  return randomArray;
  }
  
  Counter<Integer> createSelectFeatureVector(Set<Integer> arg1Type, Set<Integer> arg2Type, 
		  Set<Integer> relLabelSet, Boolean isSingleYlabel, int singleYlabel){
	  
	  Counter<Integer> yFeats_select = new ClassicCounter<Integer>();
	  
	  if(isSingleYlabel){
		  for(int type2 : arg2Type){
			  for(int type1 : arg1Type){
				  int key = singleYlabel + (type1 * labelIndex.size()) + (type2 * labelIndex.size() * argTypeIndex.size());
				  yFeats_select.incrementCount(key);
			  }
		  }
		  
		  for(int typ : arg1Type){
			  int key = singleYlabel + (typ * labelIndex.size());
			  yFeats_select.incrementCount(key);
		  }
		  
		  for(int typ : arg2Type){
			  int key = singleYlabel + (typ * labelIndex.size() * argTypeIndex.size());
			  yFeats_select.incrementCount(key);
		  }
	  }
	  
	  else {
		  for(int label : relLabelSet){
			  for(int type2 : arg2Type){
				  for(int type1 : arg1Type){
					  int key = label + (type1 * labelIndex.size()) + (type2 * labelIndex.size() * argTypeIndex.size());
					  yFeats_select.incrementCount(key);
				  }
			  }
		  }
		  
		  for(int label : relLabelSet){
			  for(int typ : arg1Type){
				  int key = label + (typ * labelIndex.size());
				  yFeats_select.incrementCount(key);
			  }
			  for(int typ : arg2Type){
				  int key = label + (typ * labelIndex.size() * argTypeIndex.size());
				  yFeats_select.incrementCount(key);
			  }  
		  }  
	  }
	  
	  return  yFeats_select;
	  
	  //TODO: check if the feature index mapping is unique with some examples
  }
  
  // TODO: This function needs relooking in light of new mentionFeatureVector
  Counter<Integer> ComputePrY_ZiTi(int [] zPredicted, Set<Integer> arg1Type, 
		  Set<Integer> arg2Type, Set<Integer> goldPosY){
	  
	  Counter<Integer> yPredicted = new ClassicCounter<Integer>();
	  
	  Set<Integer> zPredictedSet[] = ErasureUtils.uncheckedCast(new Set[zPredicted.length]);
	  for(int i = 0; i < zPredicted.length; i++){
		  zPredictedSet[i] = new HashSet<Integer>();
		  zPredictedSet[i].add(zPredicted[i]);
	  }
	  
	  Counter<Integer> yFeats_selectNil = createSelectFeatureVector(arg1Type, arg2Type, goldPosY, true, nilIndex);
	  double scoreSelNil = selectFweights.dotProduct(yFeats_selectNil);
	  //scoreSelNil = Math.exp(scoreSelNil);
	  
	  Counter<Integer> yFeats_mentionNil = createMentionFeatureVector(goldPosY, zPredictedSet, -1, -1, true, nilIndex);
	  double scoreMenNil = mentionFweights.dotProduct(yFeats_mentionNil);
	  //scoreMenNil = Math.exp(scoreMenNil);
	  
	  double scoreNil = scoreSelNil + scoreMenNil;
	  
	  for(String labelName : labelIndex){
		  int yLabel = labelIndex.indexOf(labelName);
		  
		  if(yLabel == nilIndex)
			  continue;
			  
		  Counter<Integer> yFeats_select = createSelectFeatureVector(arg1Type, arg2Type, goldPosY, true, yLabel);
		  double scoreSel = selectFweights.dotProduct(yFeats_select);
		  //scoreSel = Math.exp(scoreSel);
		  
		  Counter<Integer> yFeats_mention = createMentionFeatureVector(goldPosY, zPredictedSet, -1, -1, true, yLabel);
		  double scoreMen = mentionFweights.dotProduct(yFeats_mention);
		  //scoreMen = Math.exp(scoreMen);
		
		  double scoreLabel = scoreSel + scoreMen;
		  
		  if(scoreLabel > scoreNil)
			  yPredicted.setCount(yLabel, 1);
	  }
	  
	  return yPredicted;
  }
  
  Counter<Integer> generateYPredicted(Counter<Integer> ys, double threshold) {
	  Counter<Integer> yPredicted = new ClassicCounter<Integer>();
	  
	  for(String label : labelIndex){
		  int indx = labelIndex.indexOf(label);
		  
		  double score = ys.getCount(indx);
		  if(score > threshold) 
			  yPredicted.setCount(indx, 1);
	  }
	  
	  return yPredicted;
  }
  
  void generateYZTPredicted(List<Counter<Integer>> ys, List<Counter<Integer>> zs, List<Counter<Integer>> ts, 
		  Counter<Integer> yPredicted, int [] zPredicted, int [] tPredicted) {
	  
  }
  List<Counter<Integer>> ComputePrZ(int [][] crtGroup) {
	  List<Counter<Integer>> prZs = estimateZ(crtGroup);
	  
	  for(Counter<Integer> pr_z : prZs){
		  
//		  System.out.println("Values before : ");
//		  for(double score : pr_z.values())
//			  System.out.print(score + " ");
//		  System.out.println();
		  
		  double scoreTotal = 0.0;
		  for(double score : pr_z.values())
			  scoreTotal += Math.exp(score);
		  
		  for(int z : pr_z.keySet()){
			  double score = Math.exp(pr_z.getCount(z));
			  pr_z.setCount(z, score/scoreTotal);
		  }
		  
//		  System.out.println("Keyset Size " + pr_z.keySet().size());
//		  System.out.println("Values after : ");
//		  for(double score : pr_z.values())
//			  System.out.print(score + " ");
//		  System.out.println();
			  
	  }
	  
	  return prZs;
  }
  
  private void trainJointly(
		  int [][] crtGroup,
          Set<Integer> goldPos,
          Set<Integer> arg1Type,
          Set<Integer> arg2Type,
          Counter<Integer> posUpdateStats,
          Counter<Integer> negUpdateStats,
          int egId,
          int epoch) {
	  
	  int [] zPredicted = null;
	  int [] tPredicted = null;
	  
	  if(ALGO_TYPE == 1){
		  // TODO: Also at a later stage, do we need to change this to block gibbs sampling to do joint inf... Pr(Y,Z | T) ?
		  // \hat{Y,Z} = argmax_{Y,Z} Pr_{\theta} (Y, Z | T_i, x_i)
		  // 1. estimate Pr(Z) .. for now estimating \hat{Z}
		  List<Counter<Integer>> pr_z = estimateZ(crtGroup); // TODO: Now calling the estimateZ function. Need to check if it has to be replaced by ComputePrZ()
		  zPredicted = generateZPredicted(pr_z); 
		  // 2. estimate Pr(Y|Z,T)
		  Counter<Integer> yPredicted = ComputePrY_ZiTi(zPredicted, arg1Type, arg2Type, goldPos);
		  //Counter<Integer> yPredicted = generateYPredicted(pr_y, 0.01); // TODO: temporary hack .. need to parameterize this
		  
		  Set<Integer> [] zUpdate;
		  
		  if(updateCondition(yPredicted.keySet(), goldPos)){
			  //TODO: Do we need to differentiate between nil labels and non-nil labels (as in updateZModel) ? Verify during small dataset runs
			  zUpdate = generateZUpdate(goldPos, crtGroup);
			  updateZModel(zUpdate, zPredicted, crtGroup, epoch, posUpdateStats, negUpdateStats);
			  updateMentionWeights(zUpdate, zPredicted, goldPos, yPredicted, epoch, posUpdateStats, negUpdateStats);
			  updateSelectWeights(goldPos, yPredicted, arg1Type, arg2Type, epoch, posUpdateStats, negUpdateStats);
		  }
//		  else {
//			  if(yPredicted.keySet().size() > 1){
//				  System.out.println("-----------------");
//				  System.out.println("Ypred : " + yPredicted.keySet());
//				  System.out.println("Gold Pos" + goldPos);
//				  System.out.println("Epoch : " + epoch);
//			  }
//		  }
	  }
	  
	  else if(ALGO_TYPE == 2){ // TODO: Need to complete this ...
		  List<Counter<Integer>> pr_y = null;
		  List<Counter<Integer>> pr_z = null;
		  List<Counter<Integer>> pr_t = null;
		  
		  Counter<Integer> yPredicted = null;
		  
		  Set<Integer> [] zUpdate;
		  Set<Integer> [] tUpdate;
		  
		  ComputePrYZT(pr_y, pr_z, pr_t);
		  generateYZTPredicted(pr_y, pr_z, pr_t, yPredicted, zPredicted, tPredicted);
		  
		  if(updateCondition(yPredicted.keySet(), goldPos)){
			  
			  //zUpdate = generateZUpdate(goldPos, pr_z);
			  zUpdate = generateZUpdate(goldPos, crtGroup);
			  tUpdate = generateTUpdate(goldPos, pr_z);
			  updateZModel(zUpdate, zPredicted, crtGroup, epoch, posUpdateStats, negUpdateStats);
		  } 
		  
	  }
	  
  }
  
  private void updateSelectWeights(Set<Integer> goldPos,
		Counter<Integer> yPredicted, Set<Integer> arg1Type,
		Set<Integer> arg2Type, int epoch, Counter<Integer> posUpdateStats,
		Counter<Integer> negUpdateStats) {
	  
	  Counter<Integer> selectFeatsToAdd = createSelectFeatureVector(arg1Type, arg2Type, goldPos, false, -1);
	  
	  Set<Integer> yPredictedSet = new HashSet<Integer>();
	  for(int yPred : yPredicted.keySet())
		  yPredictedSet.add(yPred);
	  Counter<Integer> selectFeatsToSub = createSelectFeatureVector(arg1Type, arg2Type, yPredictedSet, false, -1);
	  
	  for (int feature : selectFeatsToAdd.keySet()){
		  selectFweights.weights[feature] += getEta(epoch);
	  }

	  for (int feature : selectFeatsToSub.keySet()){
		  selectFweights.weights[feature] -= getEta(epoch);
	  }
	
//	  for(double w : selectFweights.weights)
//			System.out.print(w + " ");
//		System.out.println();
}
  
private void updateMentionWeights(Set<Integer>[] zUpdate, int[] zPredicted,
		Set<Integer> goldPosY, Counter<Integer> yPredicted, int epoch,
		Counter<Integer> posUpdateStats, Counter<Integer> negUpdateStats) {
	
	Counter<Integer> mentionFVtoadd = createMentionFeatureVector(goldPosY, zUpdate, -1, -1, false, -1);

	Set<Integer> yPredictedSet = new HashSet<Integer>();
	for(int y : yPredicted.keySet()){
		yPredictedSet.add(y);
	}
	
	Set<Integer> zPredictedSet[] = ErasureUtils.uncheckedCast(new Set[zPredicted.length]);
	for(int i = 0; i < zPredicted.length; i++){
		  zPredictedSet[i] = new HashSet<Integer>();
		  zPredictedSet[i].add(zPredicted[i]);
	}
	Counter<Integer> mentionFVtosub = createMentionFeatureVector(yPredictedSet, zPredictedSet, -1, -1, false, -1);
	
	for (int feature : mentionFVtoadd.keySet()){
		mentionFweights.weights[feature] += getEta(epoch);
	}

	for (int feature : mentionFVtosub.keySet()){
		mentionFweights.weights[feature] -= getEta(epoch);
	}
	
	//TODO: To use posUpdateStats and negUpdateStats
//	for(double w : mentionFweights.weights)
//		System.out.print(w + " ");
//	System.out.println();
	
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
  
  //TODO: Why is nilLabel treated separately ? Do we have to do the same in other update functions ? Or we have change it here ?
  private void updateZModel(
          Set<Integer> [] goldZ,
          int [] predictedZ,
          int [][] group,
          int epoch,
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
        //zWeights[pred].update(datum, -1.0);
        zWeights[pred].update(datum, -getEta(epoch)); // changing the learning rate
        negUpdateStats.incrementCount(pred);
        negUpdateStats.incrementCount(LABEL_ALL);
      }

      // negative update for NIL
      if(pred == nilIndex && gold.size() != 0){
        zWeights[nilIndex].update(datum, -getEta(epoch)); // changing the learning rate
        negUpdateStats.incrementCount(pred);
      }

      // positive update(s)
      for(int l: gold) {
        if(l != nilIndex && l != pred) {
          zWeights[l].update(datum, +getEta(epoch)); // changing the learning rate
          posUpdateStats.incrementCount(l);
          posUpdateStats.incrementCount(LABEL_ALL);
        }
      }

      // positive update for NIL
      if(gold.size() == 0 && pred != nilIndex){
        zWeights[nilIndex].update(datum, +getEta(epoch)); // changing the learning rate
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
  
  Counter<Integer> createVector(Set<Integer> setRepresentation){ 
	  Counter<Integer> vector = new ClassicCounter<Integer>();

	  for(int y : setRepresentation){
		  vector.incrementCount(y);
	  }
	  
	  return vector;
  }
  
  Counter<Integer> createMentionFeatureVector(Set<Integer> yLabelsSet, Set<Integer> [] zPredictedSet, int zVar, 
		  int zLabel, Boolean isSingleYlabel, int singleYlabel){
	  
	  Counter<Integer> mentionFeatureVector = new ClassicCounter<Integer>();
	  
	  int [] zPredicted = new int[zPredictedSet.length];
	  for(int i = 0; i < zPredictedSet.length; i ++){
		  Set<Integer> zPred = zPredictedSet[i];
		  Iterator<Integer> iter = zPred.iterator();
		  zPredicted[i] = iter.next();
	  }
	  	  
	  if(isSingleYlabel){
		  for(int j = 0; j < zPredicted.length; j ++){
			  
			  int z = zPredicted[j];
			  
			  int key = (singleYlabel * labelIndex.size()) + z;
			  mentionFeatureVector.incrementCount(key);
		  }
	  }
	  
	  else {
		  Counter<Integer> yLabelsVector = createVector(yLabelsSet);
		  for(int ylabel : yLabelsVector.keySet()){
			  for(int j = 0; j < zPredicted.length; j ++){
				  
				  int zPredLabel = zPredicted[j];
				  
				  if(j == zVar){ //for the particular zVar : Pr(zVar | otherZ, Yi)
					  int key = (ylabel * labelIndex.size()) + zLabel; // add the zLabel for this zVar instead of zPredLabel
					  mentionFeatureVector.incrementCount(key);
				  }
				  else{
					  int key = (ylabel * labelIndex.size()) + zPredLabel;
					  mentionFeatureVector.incrementCount(key);
				  }
			  }
		  }
	  }
	  
	  return mentionFeatureVector;
  }

  /**
   * Gibbs sampling of Z
   * ------------------------------------------------
   * while (not converged) {
   * 	choose a random permutation for Z variable Pi
   * 	for (j = 1 to |Z|) {
   * 		update Z_{Pi_j} given the rest is fixed
   * 	}
   * }
   */
  	private Set<Integer> [] generateZUpdate(
          Set<Integer> goldPosY,
          int [][] crtGroup) {
	  
  	  List<Counter<Integer>> pr_z = ComputePrZ(crtGroup);
  	  int zPredicted [] = generateZPredicted(pr_z);
  		
	  Set<Integer> [] zUpdate = ErasureUtils.uncheckedCast(new Set[pr_z.size()]);
	  //Initialise zUpdate
	  for(int i = 0; i < zUpdate.length; i++){
		  zUpdate[i] = new HashSet<Integer>();
		  zUpdate[i].add(zPredicted[i]);
	  }
	  
	  Counter<Integer> zScoresMen = new ClassicCounter<Integer>();
	  Counter<Integer> zScoresExt = new ClassicCounter<Integer>();
	  
	  for(int i = 0; i < epochsInf; i ++){
		  //Log.severe("Gibbs Sampling: Started epochInf #" + i + "...");
		  ArrayList<Integer> randomArray = randomizeVar(pr_z.size()); 
		  
		  for(int j = 0; j < zUpdate.length; j ++){

			  int zVar = randomArray.get(j);
			  
			  double totalScoreMen = 0.0;
			  double totalScoreExt = 0.0;

			  for(int zLabel = 0; zLabel < labelIndex.size(); zLabel ++){
				  Counter<Integer> mentionFeatureVector = createMentionFeatureVector(goldPosY, zUpdate, zVar, zLabel, false, -1);  
				  Double scoreMen = mentionFweights.dotProduct(mentionFeatureVector);
				 
				  // TODO: Temporary hack to take care of +infinity
				  if(scoreMen >= 700){
					  scoreMen = Double.MAX_VALUE;
					  totalScoreMen = Double.MAX_VALUE;
				  }
				  else{
					  scoreMen = Math.exp(scoreMen);
					  totalScoreMen += scoreMen;
				  }
				  
				  zScoresMen.setCount(zLabel, scoreMen);
				  
				  //adding the extract factors for the given zVar (i.e. Xi for the given Zi and label fixed to zLabel)
				  double scoreExt = estimateZ(crtGroup[zVar], zLabel);
				  scoreExt += Math.exp(scoreExt);
				  zScoresExt.setCount(zLabel, scoreExt);
			  }
			  
			  // normalize the zScores
			  for(int zLabel : zScoresMen.keySet()){
				  double score = zScoresMen.getCount(zLabel);
				  score = score / totalScoreMen;
				  zScoresMen.setCount(zLabel, score);
			  }
			  
			  for(int zLabel : zScoresExt.keySet()){
				  double score = zScoresExt.getCount(zLabel);
				  score = score / totalScoreExt;
				  zScoresExt.setCount(zLabel, score);
			  }
			  
			  //TODO: Smoothing, in case of 0 counts; Need to investigate why 0 is is coming in totalScore; IS this RIGHT ?
			  if(totalScoreMen == 0){
				  //System.out.println("-----0 in totalScoreMen-----");
				  for(int zLabel : zScoresMen.keySet()){
					  double score = 1.0 / labelIndex.size();
					  zScoresMen.setCount(zLabel, score);
				  }
			  }
			  if(totalScoreExt == 0){
				  //System.out.println("-----0 in totalScoreExt-----");
				  for(int zLabel : zScoresExt.keySet()){
					  double score = 1.0 / labelIndex.size();
					  zScoresExt.setCount(zLabel, score);
				  }
			  }
			  
			  Set<Integer> bestZ = selectBestZ(zScoresMen, zScoresExt);
			  // add the bestZ (from the estimated scores of current set of zLabels) for variable Zj in the 'i'th iteration (also clear the old label)
			  zUpdate[zVar].clear();
			  zUpdate[zVar].addAll(bestZ); 
		  }
	  }

	  return zUpdate;
  }
  	
  Set<Integer> selectBestZ(Counter<Integer> zScoresMen, Counter<Integer> zScoresExt){
	  Set<Integer> bestZ = new HashSet<Integer>();
	  
	  double maxScore = Double.MIN_VALUE;
	  int maxZ = -1;
	  for(int z : zScoresMen.keySet()){
		  double score = zScoresMen.getCount(z);
		  
		  score *= zScoresExt.getCount(z);
		  
		  if(score > maxScore){
			  maxScore = score;
			  maxZ = z;
		  }
	  }
	  
	  if(maxZ != -1)
		  bestZ.add(maxZ);
	  else{
		  System.out.println("------------------ERROR no max in Z!! -------------------");
	  }
	  
	  return bestZ;
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

