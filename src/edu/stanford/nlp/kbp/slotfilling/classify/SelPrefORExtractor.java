package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.kbp.slotfilling.KBPEvaluator;
import edu.stanford.nlp.kbp.slotfilling.KBPTrainer;
import edu.stanford.nlp.kbp.slotfilling.classify.HoffmannExtractor.Edge;
import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.StringUtils;
import ilpInference.InferenceWrappers;
import ilpInference.YZPredicted;

import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.sri.faust.gazetteer.Gazetteer;
import com.sri.faust.gazetteer.maxmind.MaxmindGazetteer;

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

  private final int ALGO_TYPE; // 1 - for test type 1 , 2 - for test type 2 (cf. write-up,notes on google doc)
  
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
    
    double avgDotProduct(Counter<Integer> vector){
    	return dotProduct(vector, avgWeights);
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
	  //return (1.0/epoch);
	  return 1.0;
  }
  
  Index<String> labelIndex;
  Index<String> zFeatureIndex;
  Index<String> argTypeIndex;
  /** Index of the NIL label */
  int nilIndex;
  /** Number of epochs during training */
  final int epochs;
  
  final int epochsInf;

  public SelPrefORExtractor(int epochs) {
    this.epochs = epochs;
    this.epochsInf = 10;
    this.ALGO_TYPE = 1;
  }
  public SelPrefORExtractor(Properties props) {
    Log.severe("SelPrefORExtractor configured with the following properties:");
    this.epochs = PropertiesUtils.getInt(props, Props.PERCEPTRON_EPOCHS, 10);
    this.epochsInf = PropertiesUtils.getInt(props, Props.INFERENCE_EPOCHS, 10);
    this.ALGO_TYPE = PropertiesUtils.getInt(props, Props.ALGOTYPE, 1);
    Log.severe("epochs = " + epochs);
    Log.severe("Algorithm type is " + ALGO_TYPE);
    Log.severe("Inference rounds (gibbs sampling) = " + epochsInf );
  }


	// 4 sets of variables for each of the factors
	// 1) f_bias 2) f_select 3) f_mention 4) f_extract

	/*
	 * f_bias -- independent
	 * f_select (T_arg1, T_arg2, Y)
	 * f_mention (Y, Z_1 .. Z_n)
	 * f_extract -- independent -- DONE
	 */
  
  @Override
  public void train(MultiLabelDataset<String, String> dataset) {
 	    Log.severe("Training the \"Selectional Preference with Overlapping relations\" model using "
	            + dataset.featureIndex().size() + " features and "
	            + "the following labels: " + dataset.labelIndex().toString());

// 	    Index<String> tmpIndx = dataset.labelIndex();
//	    
// 	    labelIndex = new HashIndex();
//	    // add the NIL label
//	    labelIndex.add(RelationMention.UNRELATED);
//	    for(int i = tmpIndx.size()-1; i >= 0; i--)
//	    	labelIndex.add(tmpIndx.get(i));
	    
 	    labelIndex = dataset.labelIndex();
 	    labelIndex.add(RelationMention.UNRELATED);
 	   
//	    System.out.println(labelIndex);
//	    System.exit(0);
	    
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

	    Log.severe("DATASET SIZE = " + dataset.size());
	    /**
	     * Training algorithm starts here
	     */
	    // repeat for a number of epochs
	    for(int t = 0; t < epochs; t ++){
	    	// randomize the data set in each epoch
	    	// use a fixed seed for replicability
	    	Log.severe("Started epoch #" + t + "...");
	    	dataset.randomize(t); // For uniformity with hoffmann algo randomization

	    	// TODO: Check -- Need to update some statistics ??
	    	Counter<Integer> posUpdateStats = new ClassicCounter<Integer>();
	    	Counter<Integer> negUpdateStats = new ClassicCounter<Integer>();

	    	// traverse the relation dataset
	    	for(int i = 0; i < dataset.size(); i ++){
	    		int [][] crtGroup = dataset.getDataArray()[i];
	    		Set<Integer> goldPos = dataset.getPositiveLabelsArray()[i];

	    		Set<Integer> arg1Type = dataset.arg1TypeArray()[i];
	    		Set<Integer> arg2Type = dataset.arg2TypeArray()[i];

	    		trainJointly(crtGroup, goldPos, arg1Type, arg2Type, posUpdateStats, negUpdateStats, i, t+1);

	    		// update the number of iterations an weight vector has survived
	    		for(LabelWeights zw: zWeights) zw.updateSurvivalIterations();
	    		
	    		//mentionFweights.updateSurvivalIterations();
	    		//selectFweights.updateSurvivalIterations();
	    	}

	    	Log.severe("Epoch #" + t + " completed. Inspected " +
	    			dataset.size() + " datum groups. Performed " +
	    			posUpdateStats.getCount(LABEL_ALL) + " ++ updates and " +
	    			negUpdateStats.getCount(LABEL_ALL) + " -- updates.");
	    	
	    }

	    // finalize learning: add the last vector to the avg for each label
	    for(LabelWeights zw: zWeights) zw.addToAverage();
	    //mentionFweights.addToAverage();
	    //selectFweights.addToAverage();
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
  
  /**
   * Load the learnt models ... for dubugging purposes
 * @throws IOException 
 * @throws ClassNotFoundException 
   */
  public static void loadModels(Properties props) throws ClassNotFoundException, IOException{
	  PrintStream os = new PrintStream("/home/ajay/Desktop/weights.txt");
	  //String modelPath1 = "corpora/kbp/kbp_relation_model_selprefor.SelPrefOR_EXTRACTOR.5.ser";
//	  SelPrefORExtractor selpreforExt = (SelPrefORExtractor)SelPrefORExtractor.load(modelPath1, props);
//	  Log.severe("Loaded SelPrefORExtractor - learnt before");
//	  os.println("selpreforExt zWeights\n-----------");
//	  for(LabelWeights z : selpreforExt.zWeights){
//		  double l1norm = 0.0;
//		  double l2norm = 0.0;
//		  for(double w : z.avgWeights){
//			  l1norm += w;
//			  l2norm += (w*w);
////			  if(w > 0)
////				  System.out.println("wt : " + w);
//		  }
//		  os.println("L1 norm : " + l1norm + " -- L2 norm : " + Math.sqrt(l2norm));
//	  }
	  
	  System.out.println("----------------------");
	  
	  String modelPath2 = "/home/ajay/Desktop/selprefor_2_100_exp_run/kbp_relation_model_selprefor.SelPrefOR_EXTRACTOR.5.ser";
	  SelPrefORExtractor  selpreforExtNew = (SelPrefORExtractor)SelPrefORExtractor.load(modelPath2, props);
	  Log.severe("Loaded SelPrefORExtractor - newly learnt");
	  os.println("SelPrefORExtractor zWeights - newly learnt\n-----------");
	  for(SelPrefORExtractor.LabelWeights z : selpreforExtNew.zWeights){
		  double l1norm = 0.0;
		  double l2norm = 0.0;
		  for(double w : z.avgWeights){
			  l1norm += w;
			  l2norm += (w*w);
//			  if(w > 0)
//				  System.out.println("wt : " + w);
		  }
		  os.println("L1 norm : " + l1norm + " -- L2 norm : " + Math.sqrt(l2norm));
	  }
	  os.println("SelPrefORExtractor mentionFweights - newly learnt( size = " + selpreforExtNew.mentionFweights.avgWeights.length +" ) \n-----------");
	  for(int i = 0; i < selpreforExtNew.mentionFweights.avgWeights.length; i++ )
		  if(selpreforExtNew.mentionFweights.avgWeights[i] > 0)
			  os.print(selpreforExtNew.mentionFweights.avgWeights[i] + " ");
	  os.println();
	  
	  os.println("SelPrefORExtractor selectFweights - newly learnt ( size = " + selpreforExtNew.selectFweights.avgWeights.length +" ) \n-----------");
	  for(int i = 0; i < selpreforExtNew.selectFweights.avgWeights.length; i++ )
		  if(selpreforExtNew.selectFweights.avgWeights[i] > 0)
		  os.print(selpreforExtNew.selectFweights.avgWeights[i] + " ");
	  os.println();
	  
	  //os.println(selpreforExt.zWeights.length);
	  //os.println(hoffmannExt.zWeights.length);
	  os.close();
	  System.exit(0);
  }
  
  static void featureVectorTest(Properties props) throws ClassNotFoundException, IOException{
	  String modelPath = "";
	  SelPrefORExtractor  selpreforExtNew = (SelPrefORExtractor)SelPrefORExtractor.load(modelPath, props);
	  Log.severe("Loaded SelPrefORExtractor - newly learnt");
	  Index<String> labelIndex = selpreforExtNew.labelIndex;
	  Counter<Integer> selectFvector;
	  Counter<Integer> mentionFvector;
	  
  }
  
  public static void main(String[] args) throws Exception{

	  Properties props = StringUtils.argsToProperties(args);
	  Log.setLevel(Log.stringToLevel(props.getProperty(Props.LOG_LEVEL)));
	  Log.severe("--------------Running the new algo (ILP inference) ---- One small step for man .... :-) ");
	  Log.severe("Using run id: " + props.getProperty(Props.RUN_ID) + " in working directory " + props.getProperty(Props.WORK_DIR));

	  //testRandomArrayGen();
	  //loadModels(props);
	  //featureVectorTest(props);
	  /**
	   * Training algorithm
	   */
	  train(props);
	  //System.exit(0);
	   
	  /**
	   * Evaluation
	   */
	  // enable coref during testing!
	  props.setProperty(Props.INDEX_PIPELINE_METHOD, "FULL");
	  // we do not care about model combination mode here
	  props.setProperty(Props.MODEL_COMBINATION_ENABLED, "false");
	  KBPEvaluator.extractAndScore(props, false);

  }

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
  
  private double estimateZ(int [] datum, int label) {
	    Counter<Integer> vector = new ClassicCounter<Integer>();
	    for(int d: datum) 
	    	vector.incrementCount(d);
	    
	    double score = zWeights[label].dotProduct(vector);
	      
	    return score;
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
  
  Counter<Integer> generateYPredicted(int [] zPredicted, Set<Integer> arg1Type, 
		  Set<Integer> arg2Type, Set<Integer> goldPosY, boolean isTrain){
	  
	  Counter<Integer> yPredicted = new ClassicCounter<Integer>();
	  
	  Set<Integer> zPredictedSet[] = ErasureUtils.uncheckedCast(new Set[zPredicted.length]);
	  for(int i = 0; i < zPredicted.length; i++){
		  zPredictedSet[i] = new HashSet<Integer>();
		  zPredictedSet[i].add(zPredicted[i]);
	  }
	  
	  Counter<Integer> yFeats_selectNil = createSelectFeatureVector(arg1Type, arg2Type, goldPosY, true, nilIndex);
	  double scoreSelNil;
	  if(isTrain)
		  scoreSelNil = selectFweights.dotProduct(yFeats_selectNil);
	  else
		  scoreSelNil = selectFweights.avgDotProduct(yFeats_selectNil);
	  //scoreSelNil = Math.exp(scoreSelNil);
	  
	  Counter<Integer> yFeats_mentionNil = createMentionFeatureVector(goldPosY, zPredictedSet, -1, -1, true, nilIndex);
	  double scoreMenNil;
	  if(isTrain)
		  scoreMenNil = mentionFweights.dotProduct(yFeats_mentionNil);
	  else
		  scoreMenNil = mentionFweights.avgDotProduct(yFeats_mentionNil);
	  //scoreMenNil = Math.exp(scoreMenNil);
	  
	  double scoreNil = scoreSelNil + scoreMenNil;
	  
	  for(String labelName : labelIndex){
		  int yLabel = labelIndex.indexOf(labelName);
		  
		  if(yLabel == nilIndex)
			  continue;
			  
		  Counter<Integer> yFeats_select = createSelectFeatureVector(arg1Type, arg2Type, goldPosY, true, yLabel);
		  double scoreSel;
		  if(isTrain)
			  scoreSel = selectFweights.dotProduct(yFeats_select);
		  else
			  scoreSel = selectFweights.avgDotProduct(yFeats_select);
		  
		  //scoreSel = Math.exp(scoreSel);
		  
		  Counter<Integer> yFeats_mention = createMentionFeatureVector(goldPosY, zPredictedSet, -1, -1, true, yLabel);
		  double scoreMen;
		  if(isTrain)
			  scoreMen = mentionFweights.dotProduct(yFeats_mention);
		  else
			  scoreMen = mentionFweights.avgDotProduct(yFeats_mention);
		  //scoreMen = Math.exp(scoreMen);
		
		  double scoreLabel = scoreSel + scoreMen;
		  
		  if(scoreLabel > scoreNil)
			  yPredicted.setCount(yLabel, Math.exp(scoreLabel));
	  }
	  
	  return yPredicted;
  }
  
//  Counter<Integer> generateYPredicted(Counter<Integer> ys, double threshold) {
//	  Counter<Integer> yPredicted = new ClassicCounter<Integer>();
//	  
//	  for(String label : labelIndex){
//		  int indx = labelIndex.indexOf(label);
//		  
//		  double score = ys.getCount(indx);
//		  if(score > threshold) 
//			  yPredicted.setCount(indx, 1);
//	  }
//	  
//	  return yPredicted;
//  }
 
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
  
  List<Counter<Integer>> computeScoresInf(List<Collection<String>> crtGroup){
	  // scores for all mentions Xj (j \in numMentions) taking different labels i \in R
	  List<Counter<Integer>> scores = new ArrayList<Counter<Integer>>();
	  
	  for(int i = 0; i < crtGroup.size(); i ++) {
		  // Xj = features of mention j; Calculating scores for Xj taking different labels i \in R = Sj_i
		  Collection<String> mention = crtGroup.get(i);
	      scores.add(calScoreInf(mention));
	  }
	  
	  return scores;
  }
  
  private Counter<Integer> calScoreInf(Collection<String> mentionFeatures) {
	  
	  	//mentionFeatures = Xj (i.e. features of the 'j'th mention)
	  	Counter<Integer> vector = new ClassicCounter<Integer>();
	    for(String feat: mentionFeatures) {
	      int idx = zFeatureIndex.indexOf(feat);
	      if(idx >= 0) vector.incrementCount(idx);
	    }

	    Counter<Integer> scores = new ClassicCounter<Integer>();
	    
	    for(int zLabel = 0; zLabel < zWeights.length; zLabel ++){
	    	// zLabel = i
	    	// score of Xj taking on label i = Sj_i
	    	double score = zWeights[zLabel].avgDotProduct(vector);
	    	scores.setCount(zLabel, score);	
	    }
	    
	    return scores;
  }
  
  private Counter<Integer> calScore(int [] mentionFeatures, Set<Integer> goldPos) {
	  
	  	//mentionFeatures = Xj (i.e. features of the 'j'th mention)
	    Counter<Integer> vector = new ClassicCounter<Integer>();
	    for(int d: mentionFeatures) vector.incrementCount(d);

	    Counter<Integer> scores = new ClassicCounter<Integer>();
	    
	    for(int zLabel = 0; zLabel < zWeights.length; zLabel ++){
	    	// zLabel = i
	    	if(goldPos == null){
	    		// score of Xj taking on label i = Sj_i
	    		double score = zWeights[zLabel].dotProduct(vector);
	    		scores.setCount(zLabel, score);
	    	}
	    	else {
	    		// Calculating scores for the inference of Pr (Z | Y,X) i.e labels goldPos = Y are given
    			// objective is Max { \Sum_{i \in Y'} \Sum_j s_ji z_ji }
    			// where Y' = set of  all rel. labels for a given mention j to be true. (i.e goldPos)
	    		
	    		if(goldPos.contains(zLabel)){ 
	    			// score of Xj taking on label i \in Y' = Sj_i
	    			double score = zWeights[zLabel].dotProduct(vector);
		    		scores.setCount(zLabel, score);
	    		}
	    	}
	    }
	    
		if(goldPos != null){ //Also calculate the nil score when goldPos is given
			int zLabel = nilIndex;
			// score of Xj taking on label i \in Y' = Sj_i
			double score = zWeights[zLabel].dotProduct(vector);
    		scores.setCount(zLabel, score);
		}

	    return scores;
  }
  
  List<Counter<Integer>> computeScores(int [][] crtGroup, Set<Integer> goldPos){
	  // scores for all mentions Xj (j \in numMentions) taking different labels i \in R
	  List<Counter<Integer>> scores = new ArrayList<Counter<Integer>>();
	  
	  for(int [] mention: crtGroup) {
		  // Xj = features of mention j; Calculating scores for Xj taking different labels i \in R = Sj_i
	      scores.add(calScore(mention, goldPos));
	  }
	  
	  return scores;
  }
  
  Counter<Integer> computeTypeBiasScores(Set<Integer> arg1Type, Set<Integer> arg2Type, boolean isInf){
	  Counter<Integer> typeBiasScores = new ClassicCounter<Integer>();
	  
	  // Wi_0
	  Counter<Integer> selectFeatureVectorNil = createSelectFeatureVector(arg1Type, arg2Type, null, true, nilIndex);
	  double scoreNil;
	  if(isInf)
		  scoreNil = selectFweights.avgDotProduct(selectFeatureVectorNil);
	  else
		  scoreNil = selectFweights.dotProduct(selectFeatureVectorNil);
	  
	  for(String yLabel : labelIndex){
		  int y = labelIndex.indexOf(yLabel);
		  
		  if(y == nilIndex)
			  continue;
		  
		  // Wi_1
		  Counter<Integer> selectFeatureVector = createSelectFeatureVector(arg1Type, arg2Type, null, true, y);
		  double score;
		  if(isInf)
			  score = selectFweights.avgDotProduct(selectFeatureVector);
		  else
			  score = selectFweights.dotProduct(selectFeatureVector);
		  
		  // Wi_1 - Wi_0 for a given 'i' = y (where i is the labelIndex of relation y)
		  typeBiasScores.setCount(y, score-scoreNil);
		  
	  }
	  
	  return typeBiasScores;
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
	  
	  if(egId % 10000 == 0)
		  Log.severe("Completed processing " + egId + " pairs of entities. ilp inference (huffmann simulation)...");
	  
	  if(ALGO_TYPE == 1){
		  // TODO: Also at a later stage, do we need to change this to block gibbs sampling to do joint inf... Pr(Y,Z | T) ?
		  //List<Counter<Integer>> pr_z = estimateZ(crtGroup); //To be replaced by ComputePrZ() ?
		  //zPredicted = generateZPredicted(pr_z); 
		  //Counter<Integer> yPredicted = generateYPredicted(zPredicted, arg1Type, arg2Type, goldPos, true);
		  List<Counter<Integer>> scores = computeScores(crtGroup, null);
		  Counter<Integer> typeBiasScores = null; //computeTypeBiasScores(arg1Type, arg2Type, false);
		  InferenceWrappers ilpInfHandle = new InferenceWrappers();
		  //Counter<Integer> yPredicted = ilpInfHandle.generateYZPredictedILP(scores, crtGroup.length, labelIndex, typeBiasScores, zPredicted);
		  YZPredicted predictedVals = ilpInfHandle.generateYZPredictedILP(scores, crtGroup.length, labelIndex, typeBiasScores, egId, epoch, nilIndex);
		  Counter<Integer> yPredicted = predictedVals.getYPredicted();
		  zPredicted = predictedVals.getZPredicted();
		  Set<Integer> [] zUpdate;
		  
//		  System.out.print("epoch: " + (epoch-1) +  "; egid: " + egId + "; z=[");
//		  for(int z : zPredicted){
//			  System.out.print(z + " ");
//		  }
//		  System.out.print("];");
//		  System.out.print(" y={");
//		  for(int y : yPredicted.keySet())
//			  System.out.print(y + " ");
//		  System.out.println("}");
		  
		  if(updateCondition(yPredicted.keySet(), goldPos)){
			  //TODO: Do we need to differentiate between nil labels and non-nil labels (as in updateZModel) ? Verify during small dataset runs
			  //zUpdate = generateZUpdate(goldPos, crtGroup);
			  
//			  if(goldPos.size() - crtGroup.length > 0){
//				  System.out.println("How come ? " + "data : " + crtGroup + " goldPos : " + goldPos + " EgId : " + egId + " ... skipping ...");
//				  return;
//			  }
			  
			  List<Counter<Integer>> scoresWithYgiven = computeScores(crtGroup, goldPos);
			  zUpdate = ilpInfHandle.generateZUpdateILP(scoresWithYgiven, crtGroup.length, goldPos, nilIndex);
			  //zUpdate = generateZUpdate(goldPos, scores);
			  updateZModel(zUpdate, zPredicted, crtGroup, epoch, posUpdateStats, negUpdateStats);
			  //updateMentionWeights(zUpdate, zPredicted, goldPos, yPredicted, epoch, posUpdateStats, negUpdateStats);
			  //updateSelectWeights(goldPos, yPredicted, arg1Type, arg2Type, epoch, posUpdateStats, negUpdateStats);
		  }
	  }
	  
	  else if(ALGO_TYPE == 2){ // TODO: Need to complete this ...
		  Counter<Integer> yPredicted = null;
		  Set<Integer> [] zUpdate;
		  
		  //1.0) \hat{Y,Z,T} = argmax Pr (Y, Z, T | Xi; \theta)
		  
		  if(updateCondition(yPredicted.keySet(), goldPos)){
			  // 2.0) Z*, T* = argmax Pr (Z, T | Yi, Xi; \theta)
			  
			  // 2.1) Pr (Z | Yi; \theta) --> gibbs sampling
			  zUpdate = generateZUpdate(goldPos, crtGroup);
			  
			  // 2.2) Pr (T | Yi; \theta) --> gibbs sampling
			  
			  
			  // 3.0) Update Weights

		  } 
		  
	  }
	  
  }
  
  /** The conditional inference from (Hoffmann et al., 2011) */
  private Set<Integer> [] generateZUpdate(
          Set<Integer> goldPos,
          List<Counter<Integer>> zs) {
    Set<Integer> [] zUpdate = ErasureUtils.uncheckedCast(new Set[zs.size()]);
    for(int i = 0; i < zUpdate.length; i ++)
      zUpdate[i] = new HashSet<Integer>();

    // build all edges, for NIL + gold labels
    List<Edge> edges = new ArrayList<Edge>();
    for(int m = 0; m < zs.size(); m ++) {
      for(Integer y: zs.get(m).keySet()) {
        if(goldPos.contains(y) || y == nilIndex) {
          double s = zs.get(m).getCount(y);
          edges.add(new Edge(m, y, s));
        }
      }
    }

    // there are more Ys than mentions
    if(goldPos.size() > zs.size()) {
      // sort in descending order of scores
      Collections.sort(edges, new Comparator<Edge>() {
        @Override
        public int compare(Edge o1, Edge o2) {
          if(o1.score > o2.score) return -1;
          else if(o1.score == o2.score) return 0;
          return 1;
        }
      });

      // traverse edges and cover as many Ys as possible
      Set<Integer> coveredYs = new HashSet<Integer>();
      for(Edge e: edges) {
        if(e.y == nilIndex) continue;
        if(! coveredYs.contains(e.y) && zUpdate[e.mention].size() == 0) {
          zUpdate[e.mention].add(e.y);
          coveredYs.add(e.y);
        }
      }

      return zUpdate;
    }

    // there are more mentions than relations

    // for each Y, pick the highest edge from an unmapped mention
    Map<Integer, List<Edge>> edgesByY = byY(edges);
    for(Integer y: goldPos) {
      List<Edge> es = edgesByY.get(y);
      assert(es != null);
      for(Edge e: es) {
        if(zUpdate[e.mention].size() == 0) {
          zUpdate[e.mention].add(e.y);
          break;
        }
      }
    }

    // map the leftover mentions to their highest scoring Y
    Map<Integer, List<Edge>> edgesByZ = byZ(edges);
    for(int m = 0; m < zUpdate.length; m ++) {
      if(zUpdate[m].size() == 0) {
        List<Edge> es = edgesByZ.get(m);
        assert(es != null);
        assert(es.size() > 0);
        if(nilIndex != es.get(0).y) {
          zUpdate[m].add(es.get(0).y);
        }
      }
    }

    return zUpdate;
  }
  
  Map<Integer, List<Edge>> byY(List<Edge> edges) {
	    Map<Integer, List<Edge>> edgesByY = new HashMap<Integer, List<Edge>>();
	    for(Edge e: edges) {
	      if(e.y == nilIndex) continue;
	      List<Edge> yEdges = edgesByY.get(e.y);
	      if(yEdges == null) {
	        yEdges = new ArrayList<Edge>();
	        edgesByY.put(e.y, yEdges);
	      }
	      yEdges.add(e);
	    }
	    for(Integer y: edgesByY.keySet()) {
	      List<Edge> es = edgesByY.get(y);
	      Collections.sort(es, new Comparator<Edge>() {
	        @Override
	        public int compare(Edge o1, Edge o2) {
	          if(o1.score > o2.score) return -1;
	          else if(o1.score == o2.score) return 0;
	          return 1;
	        }
	      });
	    }
	    return edgesByY;
	  }

	  Map<Integer, List<Edge>> byZ(List<Edge> edges) {
	    Map<Integer, List<Edge>> edgesByZ = new HashMap<Integer, List<Edge>>();
	    for(Edge e: edges) {
	      List<Edge> mentionEdges = edgesByZ.get(e.mention);
	      if(mentionEdges == null) {
	        mentionEdges = new ArrayList<Edge>();
	        edgesByZ.put(e.mention, mentionEdges);
	      }
	      mentionEdges.add(e);
	    }
	    for(Integer m: edgesByZ.keySet()) {
	      List<Edge> es = edgesByZ.get(m);
	      Collections.sort(es, new Comparator<Edge>() {
	        @Override
	        public int compare(Edge o1, Edge o2) {
	          if(o1.score > o2.score) return -1;
	          else if(o1.score == o2.score) return 0;
	          return 1;
	        }
	      });
	    }
	    return edgesByZ;
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
	  
	  if(selectFeatsToAdd.size() > 0){
		  selectFweights.addToAverage();
		  for (int feature : selectFeatsToAdd.keySet()){
			  selectFweights.weights[feature] += getEta(epoch);
			  posUpdateStats.incrementCount(LABEL_ALL);
		  }
		  selectFweights.survivalIterations = 0;
	  }

	  if(selectFeatsToSub.size() > 0){
		  selectFweights.addToAverage();
		  for (int feature : selectFeatsToSub.keySet()){
			  selectFweights.weights[feature] -= getEta(epoch);
			  negUpdateStats.incrementCount(LABEL_ALL);
		  }
		  selectFweights.survivalIterations = 0;
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
	
	if(mentionFVtoadd.size() > 0){
		mentionFweights.addToAverage();
		for (int feature : mentionFVtoadd.keySet()){
			mentionFweights.weights[feature] += getEta(epoch);
			posUpdateStats.incrementCount(LABEL_ALL);
		}
		mentionFweights.survivalIterations = 0;
	}

	
	if(mentionFVtosub.size() > 0){
		mentionFweights.addToAverage();
		for (int feature : mentionFVtosub.keySet()){
			mentionFweights.weights[feature] -= getEta(epoch);
			negUpdateStats.incrementCount(LABEL_ALL);
		}
		mentionFweights.survivalIterations = 0;
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
		  zPredicted[i] = zPred.iterator().next();  
	  }
	  	  
	  if(isSingleYlabel){
		  for(int j = 0; j < zPredicted.length; j ++){
			  
			  int z = zPredicted[j];
			  
			  int key = (singleYlabel * labelIndex.size()) + z;
			  mentionFeatureVector.incrementCount(key);
		  }
	  }
	  
	  else {
		  Counter<Integer> yLabelsVector;
		  //yLabelsSet can be null --- to handle this
		  if(yLabelsSet.size() == 0){
			  yLabelsVector = new ClassicCounter<Integer>();
			  yLabelsVector.incrementCount(nilIndex);
			  //Log.severe("Ajay : yLabelsSet is null");
		  }
		  else
			  yLabelsVector = createVector(yLabelsSet);
		  
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
	  
  	  List<Counter<Integer>> pr_z = estimateZ(crtGroup); // ComputePrZ(crtGroup);
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
				  double scoreMen = mentionFweights.dotProduct(mentionFeatureVector);
				 
				  // TODO: Temporary hack to take care of +infinity
				  if(scoreMen >= 700){
					  //Log.severe("Ajay ---- infinity -- " + scoreMen + Math.exp(scoreMen));
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
				  scoreExt = Math.exp(scoreExt);
				  totalScoreExt += scoreExt;
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
				  //Log.severe("Ajay : -----0 in totalScoreMen-----");
				  for(int zLabel : zScoresMen.keySet()){
					  double score = 1.0 / labelIndex.size();
					  zScoresMen.setCount(zLabel, score);
				  }
			  }
			  if(totalScoreExt == 0){
				  //Log.severe("Ajay : -----0 in totalScoreExt-----");
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
	  
	  double maxScore = -1; //zScoreMen and zScoreExt contain on positive values ( of type e^{})
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
//		  for(int cntr : zScoresMen.keySet())
//			  System.out.println(cntr + " : " + zScoresMen.getCount(cntr));
//		  
//		  for(int cntr : zScoresExt.keySet())
//			  System.out.println(cntr + " : " + zScoresExt.getCount(cntr));
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

  private void extractArgTypes(List<Collection<String>> mentions, Set<Integer> arg1Type, Set<Integer> arg2Type){
	  
	  for(int i = 0; i < mentions.size(); i ++){
		  Collection<String> mentionFeatures = mentions.get(i);
		  
		  String arg1_arg2 = mentionFeatures.iterator().next();
		  String REGEX = "arg1type\\=(.*)\\:(.*)\\_and\\_arg2type\\=(.*)";
		  Pattern r = Pattern.compile(REGEX);
		  Matcher m = r.matcher(arg1_arg2);
		  if(m.find()){
			  int arg1 = argTypeIndex.indexOf(m.group(2));
		      int arg2 = argTypeIndex.indexOf(m.group(3));
		      
		      if(arg1 < 0){
		    	  Log.severe("Ajay: New arg1 type in test : " + m.group(2) + " .. adding");
		    	  argTypeIndex.add(m.group(2));
		    	  arg1 = argTypeIndex.indexOf(m.group(2));
		      }
		      
		      if(arg2 < 0){
		    	  Log.severe("Ajay: New arg2 type in test : " + m.group(3) + " .. adding");
		    	  argTypeIndex.add(m.group(3));
		    	  arg2 = argTypeIndex.indexOf(m.group(3));
		      }
		      
		      arg1Type.add(arg1);
		      arg2Type.add(arg2);
		      
//		      System.out.println(argTypeIndex);
//			  System.out.println("Found value: " + m.group(0));
//			  System.out.println("Found value: " + m.group(1));
//		      System.out.println("Found value: " + m.group(2));//  + " : " + arg1 );
//		      System.out.println("Found value: " + m.group(3));//  + " : " + arg2);
		  }
		  else{
			  Log.severe("ARG1type and ARG2type not found ... quitting");
			  System.exit(0);
		  }
		  //System.out.println("-------\narg1-arg2 : " + arg1_arg2);
	  }
  }
  
  /**
   * Our inference function
   * Currently coding Pr(Y,Z|T) equivalent to the one used in training
   */
  /*
  @Override
  public Counter<String> classifyMentions(List<Collection<String>> mentions) {
	  Counter<String> yScores = new ClassicCounter<String>();
	 
	  Set<Integer> arg1Type = new HashSet<Integer>();
	  Set<Integer> arg2Type = new HashSet<Integer>();    
	  extractArgTypes(mentions, arg1Type, arg2Type);
	  
	  List<Counter<Integer>> scoreInf = computeScoresInf(mentions);
	  Counter<Integer> typeBiasScores = null; //computeTypeBiasScores(arg1Type, arg2Type, true);
	  InferenceWrappers ilpInfHandle = new InferenceWrappers();
	  
	  YZPredicted predictedVals = ilpInfHandle.generateYZPredictedILP(scoreInf, mentions.size(), labelIndex, typeBiasScores, -1, -1);
	  Counter<Integer> yPredicted = predictedVals.getYPredicted();
	  int [] zPredicted = predictedVals.getZPredicted();
	  
	  // Compute the yScores 
	  for(int y_i : yPredicted.keySet()){
		  if(y_i == nilIndex) // The calling function mandates not to predict nil label
			  continue;
		  double yscore = 0.0; //typeBiasScores.getCount(y_i);
		  for(int j = 0; j < zPredicted.length; j ++){
			  int z = zPredicted[j];
			  if(z == y_i){
				  yscore += scoreInf.get(j).getCount(y_i);
			  }
		  }
		  yScores.setCount(labelIndex.get(y_i), yscore);
	  }
	  
	  return yScores;
  }*/
  
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
  
//  public Counter<String> classifyMentions(List<Collection<String>> mentions) {
//    Counter<String> yScores = new ClassicCounter<String>();
//
//    List<Counter<Integer>> mentionScoresList = new ArrayList<Counter<Integer>>();
//    // traverse of all mention of this tuple
//    for (int i = 0; i < mentions.size(); i++) {
//      // get all scores for this mention
//      Collection<String> mentionFeatures = mentions.get(i);
//      Counter<Integer> mentionScores = classifyMention(mentionFeatures);
//
//      mentionScoresList.add(mentionScores);
//      
//    }
//
//    int zPredicted[] = generateZPredicted(mentionScoresList);
//    
//    Set<Integer> arg1Type = new HashSet<Integer>();
//    Set<Integer> arg2Type = new HashSet<Integer>();    
//    extractArgTypes(mentions, arg1Type, arg2Type);
//    
//    boolean isTrain = false; //Inference hence the following code should call avgDotProduct instead of dotProduct
//    Counter<Integer> yPredicted = generateYPredicted(zPredicted, arg1Type, arg2Type, null, isTrain);
//    
//    for(int indx : yPredicted.keySet()){
//    	String label = labelIndex.get(indx);
//    	double value = yPredicted.getCount(indx);
//    	yScores.setCount(label, value);
//    }
//    
////    if(bestZScores.keySet().size() > 1)
////    	System.out.println("Ajay : " + bestZScores);
//    
//    return yScores;
//  }

  /*private Counter<Integer> classifyMention(Collection<String> testDatum) {
 
	/**
	 * estimateZ routine during inference. Calling avgDotProduct   
	 * /
	  
    Counter<Integer> vector = new ClassicCounter<Integer>();
    for(String feat: zFeatureIndex) {
      int idx = zFeatureIndex.indexOf(feat);
      if(idx >= 0) vector.incrementCount(idx);
    }
    
    Counter<Integer> scores = new ClassicCounter<Integer>();
    for(int labelIdx = 0; labelIdx < zWeights.length; labelIdx ++){
      double score = zWeights[labelIdx].avgDotProduct(vector);
      //score = zWeights[labelIdx].avgDotProduct(testDatum, zFeatureIndex);
      scores.setCount(labelIdx, score);
    }

    return scores;
  }*/

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

    mentionFweights.clear();
    selectFweights.clear();
    
    FileOutputStream fos = new FileOutputStream(modelPath);
    ObjectOutputStream out = new ObjectOutputStream(fos);

    assert(zWeights != null);
    out.writeInt(zWeights.length);
    for(LabelWeights zw: zWeights) {
      out.writeObject(zw);
    }

    out.writeObject(labelIndex);
    out.writeObject(zFeatureIndex);
    
    // Write out new indices and weights learnt 
    out.writeObject(argTypeIndex);
    assert(mentionFweights != null);
    out.writeObject(mentionFweights);
    assert(selectFweights != null);
    out.writeObject(selectFweights);
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
    
    // read the newly created objects
    argTypeIndex = ErasureUtils.uncheckedCast(in.readObject());
    mentionFweights = ErasureUtils.uncheckedCast(in.readObject());
    selectFweights = ErasureUtils.uncheckedCast(in.readObject());
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

