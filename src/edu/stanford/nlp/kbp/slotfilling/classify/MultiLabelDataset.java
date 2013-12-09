package edu.stanford.nlp.kbp.slotfilling.classify;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import edu.stanford.nlp.kbp.slotfilling.KBPTrainer;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;

public class MultiLabelDataset<L, F> implements Serializable {
  private static final long serialVersionUID = 1L;
  
  public Index<L> labelIndex;
  public Index<F> featureIndex;
  
  /** Stores the list of known positive labels for each datum group */
  protected Set<Integer> [] posLabels;
  /** Stores the list of known negative labels for each datum group */
  protected Set<Integer> [] negLabels;
  /** Stores the datum groups, where each group consists of a collection of datums */
  protected int[][][] data;
  protected int size;

  /*
   * Added by Ajay : 30/09/2013
   * Type of the 1st and 2nd args of each datum
   */
  public Index<L> argTypeIndex; // Ajay: new variable for argument type
  
  protected Set<Integer> [] arg1TypesArray;
  public Set<Integer>[] arg1TypeArray() {
	return arg1TypesArray;
  }

  protected Set<Integer> [] arg2TypesArray;
  public Set<Integer>[] arg2TypeArray() {
	  return arg2TypesArray;
  }
  
  public Index<L> argFeatIndex; // Ajay: new variable for suffix features of entity types
  
  public Index<L> argFeatIndex() { return argFeatIndex; }

  protected Map<Integer, Counter<Integer>> suffixFeatCounts;
  
  protected Counter<Integer> entityTypeCounts;
  

public MultiLabelDataset() {
    this(10);
  }
  
  public MultiLabelDataset(int sz) {
    initialize(sz);
  }
  
  public MultiLabelDataset(int[][][] data,
      Index<F> featureIndex,
      Index<L> labelIndex,
      Set<Integer> [] posLabels,
      Set<Integer> [] negLabels) {
    this.data = data;
    this.featureIndex = featureIndex;
    this.labelIndex = labelIndex;
    this.posLabels = posLabels;
    this.negLabels = negLabels;
    this.size = data.length;
  }
  
  @SuppressWarnings("unchecked")
  protected void initialize(int numDatums) {
    labelIndex = new HashIndex<L>();
    featureIndex = new HashIndex<F>();
    posLabels = new Set[numDatums];
    negLabels = new Set[numDatums];
    data = new int[numDatums][][];
    size = 0;
    
 // Added by Ajay : 02/10/2013
    argTypeIndex = new HashIndex<L>(); 
    arg1TypesArray = new Set[numDatums];
    arg2TypesArray = new Set[numDatums];
    argFeatIndex = new HashIndex<L>();
    suffixFeatCounts = new HashMap<Integer, Counter<Integer>>();
    entityTypeCounts = new ClassicCounter<Integer>();

  }
  
  public int size() { return size; }
  
  public Index<L> labelIndex() { return labelIndex; }

  public Index<F> featureIndex() { return featureIndex; }
  
  public Index<L> argTypeIndex() { return argTypeIndex; }

  public int numFeatures() { return featureIndex.size(); }

  public int numClasses() { return labelIndex.size(); }
  
  public Set<Integer> [] getPositiveLabelsArray() {
    posLabels = trimToSize(posLabels);
    return posLabels;
  }
  
  public Set<Integer> [] getNegativeLabelsArray() {
    negLabels = trimToSize(negLabels);
    return negLabels;
  }

  public int[][][] getDataArray() {
    data = trimToSize(data);
    return data;
  }
  
  @SuppressWarnings("unchecked")
  protected Set<Integer> [] trimToSize(Set<Integer> [] i) {
    if(i.length == size) return i;
    Set<Integer> [] newI = new Set[size];
    System.arraycopy(i, 0, newI, 0, size);
    return newI;
  }
  
  protected int[][][] trimToSize(int[][][] i) {
    if(i.length == size) return i;
    int[][][] newI = new int[size][][];
    System.arraycopy(i, 0, newI, 0, size);
    return newI;
  }
  
  /**
   * Randomizes the data array in place
   * @param randomSeed
   */
  public void randomize(int randomSeed) {
    Random rand = new Random(randomSeed);
    for(int j = size - 1; j > 0; j --){
      int randIndex = rand.nextInt(j);
      
      int [][] tmp = data[randIndex];
      data[randIndex] = data[j];
      data[j] = tmp;
      
      Set<Integer> tmpl = posLabels[randIndex];
      posLabels[randIndex] = posLabels[j];
      posLabels[j] = tmpl;
      
      tmpl = negLabels[randIndex];
      negLabels[randIndex] = negLabels[j];
      negLabels[j] = tmpl;
      
    }
  }
  
  public void randomize(int [][] zLabels, int randomSeed) {
    Random rand = new Random(randomSeed);
    for(int j = size - 1; j > 0; j --){
      int randIndex = rand.nextInt(j);
      
      int [][] tmp = data[randIndex];
      data[randIndex] = data[j];
      data[j] = tmp;
      
      Set<Integer> tmpl = posLabels[randIndex];
      posLabels[randIndex] = posLabels[j];
      posLabels[j] = tmpl;
      
      tmpl = negLabels[randIndex];
      negLabels[randIndex] = negLabels[j];
      negLabels[j] = tmpl;
      
      int [] tmpz = zLabels[randIndex];
      zLabels[randIndex] = zLabels[j];
      zLabels[j] = tmpz;
    }
  }
  
  /**
   * Get the total count (over all data instances) of each feature
   *
   * @return an array containing the counts (indexed by index)
   */
  public float[] getFeatureCounts() {
    float[] counts = new float[featureIndex.size()];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < data[i].length; j++) {
        for(int k = 0; k < data[i][j].length; k ++) {
          counts[data[i][j][k]] += 1.0;
        }
      }
    }
    return counts;
  }

  // TODO: Check: Do we need to alter the new features added (entity suffix features) here ??
  /**
   * Applies a feature count threshold to the Dataset.
   * All features that occur fewer than <i>k</i> times are expunged.
   */
  public void applyFeatureCountThreshold(int threshold) {
    float[] counts = getFeatureCounts();
    
    //
    // rebuild the feature index
    //
    Index<F> newFeatureIndex = new HashIndex<F>();
    int[] featMap = new int[featureIndex.size()];
    for (int i = 0; i < featMap.length; i++) {
      F feat = featureIndex.get(i);
      if (counts[i] >= threshold) {
        int newIndex = newFeatureIndex.size();
        newFeatureIndex.add(feat);
        featMap[i] = newIndex;
      } else {
        featMap[i] = -1;
      }
    }

    featureIndex = newFeatureIndex;

    //
    // rebuild the data
    //
    for (int i = 0; i < size; i++) {
      for(int j = 0; j < data[i].length; j ++){
        List<Integer> featList = new ArrayList<Integer>(data[i][j].length);
        for (int k = 0; k < data[i][j].length; k++) {
          if (featMap[data[i][j][k]] >= 0) {
            featList.add(featMap[data[i][j][k]]);
          }
        }
        data[i][j] = new int[featList.size()];
        for(int k = 0; k < data[i][j].length; k ++) {
          data[i][j][k] = featList.get(k);
        }
      }
    }
  }
  
  /* deprecated function --- see the modified function
   * 
  public void addDatum(Set<L> yPos, Set<L> yNeg, List<Datum<L, F>> group) {
    List<Collection<F>> features = new ArrayList<Collection<F>>();
    for(Datum<L, F> datum: group){
      features.add(datum.asFeatures());
    }
    add(yPos, yNeg, features);
  }
  */
  
  /**
   * 30/09/2013 : Ajay new addDatum routine to add entity type info of the arguments to a relation
   * 
   * @param yPos
   * @param yNeg
   * @param group
   * @param arg1Val
   * @param arg1Type
   * @param arg2Val
   * @param arg2listTypes
   */
  public void addDatum(Set<L> yPos, Set<L> yNeg, List<Datum<L, F>> group, L arg1Val, L arg1Type, L arg2Val, List<L> arg2listTypes ) {
	    List<Collection<F>> features = new ArrayList<Collection<F>>();
	    for(Datum<L, F> datum: group){
	      features.add(datum.asFeatures());
	    }
	    add(yPos, yNeg, features, arg1Val, arg1Type, arg2Val, arg2listTypes); // Modified to add arg types
	    //this.arg1_type.add(arg1);
	    //this.arg2_types.add(arg2list);
	    
//	    for(L arg2 : arg2listTypes){
//	    	if(arg2.equals("URL"))
//	    		System.out.println(arg2 + " val : " + arg2Val + " ypos : " + yPos + " yneg : " + yNeg);
//	    }
  }
  
  public void add(Set<L> yPos, Set<L> yNeg, List<Collection<F>> group, L arg1Val, L arg1Type, L arg2Val, List<L> arg2listTypes) {
	  ensureSize();

	  addPosLabels(yPos);
	  addNegLabels(yNeg);
	  addFeatures(group);

	  /*
	   * Added by ajay for arg types
	   */

	  // Treat PER--PERSON and ORG-ORGANIZATION same in the types 
	  // DONE: (temporarily handled this. Leaving the comment to check for any future normalisations.
	  
	  if(arg1Type.equals("PER"))
		  arg1Type = (L) "PERSON";
	  else if(arg1Type.equals("ORG"))
		  arg1Type = (L) "ORGANIZATION";
	  
	  
	  addArgTypes(arg1Type, arg2listTypes);
	  addArgFeatures(arg1Val, arg1Type, arg2Val, arg2listTypes);
	  
	  size ++;
  }
  
  protected void addArgFeatures(L arg1Val, L arg1Type, L arg2Val, List<L> arg2listTypes){
	  
	  int lastIndx;
	  
	  if(arg1Type.toString().equalsIgnoreCase( "DATE")){
		  // TODO: Do something different
		  
	  }
	  else {
		  
		  // TODO: Lot of reititions in code in the following blocks. Consider writing a function
		  
		  lastIndx = arg1Val.toString().length();
		  if(lastIndx - 4 < 0){
			  for(int i = 0; i < lastIndx; i++){
				  L suffString = (L) arg1Val.toString().substring(i, lastIndx);
				  argFeatIndex.add(suffString);
				  addSuffixFeat(suffString, arg1Type);
			  }
		  }
		  else {
			  for(int i = 1; i <= 4; i++){
				  L suffString =  (L) arg1Val.toString().substring(lastIndx-i, lastIndx);
				  argFeatIndex.add(suffString);
				  addSuffixFeat(suffString, arg1Type);
			  }
				  
		  }
	  }
	  
	  if(containsDate(arg2listTypes)){
		  // TODO: Do something different

	  }
	  else {

		  lastIndx = arg2Val.toString().length();
		  if(lastIndx - 4 < 0){
			  for(int i = 0; i < lastIndx; i++){
				  L suffString = (L) arg2Val.toString().substring(i, lastIndx);
				  argFeatIndex.add(suffString);
				  for(L arg2type : arg2listTypes)
					  addSuffixFeat(suffString, arg2type);
			  }
		  }
		  else {
			  for(int i = 1; i <= 4; i++){
				  L suffString = (L) arg2Val.toString().substring(lastIndx-i, lastIndx);
				  argFeatIndex.add(suffString);
				  for(L arg2type : arg2listTypes)
					  addSuffixFeat(suffString, arg2type);
			  }
		  }
	  }
	  
//	  System.out.println("Arg1-type : " + arg1Type + " Arg1-type Index : " + argTypeIndex.indexOf(arg1Type));
//	  System.out.print("Arg2-type : " + arg2listTypes + " Arg2-type Index : ");
//	  for(L l : arg2listTypes){
//		  System.out.print(argTypeIndex.indexOf(l) + ", ");
//	  }
//	  System.out.println();
	  
  }
  
  private void addSuffixFeat(L suffString, L argType){	  
	  if(suffixFeatCounts.containsKey(argFeatIndex.indexOf(suffString))){
		  	Counter<Integer> cntr = suffixFeatCounts.get(argFeatIndex.indexOf(suffString));
		  	cntr.incrementCount(argTypeIndex.indexOf(argType));
	  }
	  else {
		  	Counter<Integer> cntr = new ClassicCounter<Integer>();
		  	cntr.setCount(argTypeIndex.indexOf(argType), 1.0);
		  	suffixFeatCounts.put(argFeatIndex.indexOf(suffString), cntr);
	  }
  }
  
  protected Boolean containsDate(List<L> argTypes){
	  for(L type : argTypes){
		  if(type.toString().equalsIgnoreCase("DATE"))
			  return true;
	  }
	  return false;
  }
  
  protected void addArgTypes(L arg1, List<L> arg2list) {
	  	  
	  argTypeIndex.add(arg1);
	  argTypeIndex.addAll(arg2list);
	  
	  // Update the overall counts of the entity types
	  entityTypeCounts.incrementCount(argTypeIndex.indexOf(arg1));
	  for(L arg2 : arg2list){
		  entityTypeCounts.incrementCount(argTypeIndex.indexOf(arg2));
	  }
	  
	  Set<Integer> newLabels = new HashSet<Integer>();
	  newLabels.add(argTypeIndex.indexOf(arg1));
	  arg1TypesArray[size]= newLabels;
	  
	  newLabels = new HashSet<Integer>();
	  for(L l : arg2list){
		  newLabels.add(argTypeIndex.indexOf(l));
	  }
	  arg2TypesArray[size]= newLabels;
  }
  
  protected void addFeatures(List<Collection<F>> group) {
    int [][] groupFeatures = new int[group.size()][];
    int datumIndex = 0;
    for(Collection<F> features: group){
      int[] intFeatures = new int[features.size()];
      int j = 0;
      for (F feature : features) {
        featureIndex.add(feature);
        int index = featureIndex.indexOf(feature);
        if (index >= 0) {
          intFeatures[j] = featureIndex.indexOf(feature);
          j++;
        }
      }
      
      int [] trimmedFeatures = new int[j];
      System.arraycopy(intFeatures, 0, trimmedFeatures, 0, j);
      groupFeatures[datumIndex] = trimmedFeatures;
      datumIndex ++;
    }
    assert(datumIndex == group.size());
    data[size] = groupFeatures;
  }
  
  protected void addPosLabels(Set<L> labels) {
    labelIndex.addAll(labels);
    Set<Integer> newLabels = new HashSet<Integer>();
    for(L l: labels) {
      newLabels.add(labelIndex.indexOf(l));
    }
    posLabels[size] = newLabels;
  }
  
  protected void addNegLabels(Set<L> labels) {
    labelIndex.addAll(labels);
    Set<Integer> newLabels = new HashSet<Integer>();
    for(L l: labels) {
      newLabels.add(labelIndex.indexOf(l));
    }
    negLabels[size] = newLabels;
  }
  
  @SuppressWarnings("unchecked")
  protected void ensureSize() {
	  if (posLabels.length == size) {
		  Set<Integer> [] newLabels = new Set[size * 2];
		  System.arraycopy(posLabels, 0, newLabels, 0, size);
		  posLabels = newLabels;

		  newLabels = new Set[size * 2];
		  System.arraycopy(negLabels, 0, newLabels, 0, size);
		  negLabels = newLabels;

		  int[][][] newData = new int[size * 2][][];
		  System.arraycopy(data, 0, newData, 0, size);
		  data = newData;      

		  /*
		   * Ajay: 02/10/13: Added to accommodate type info
		   */
		  newLabels = new Set[size * 2];
		  System.arraycopy(arg1TypesArray, 0, newLabels, 0, size);
		  arg1TypesArray = newLabels;

		  newLabels = new Set[size * 2];
		  System.arraycopy(arg2TypesArray, 0, newLabels, 0, size);
		  arg2TypesArray = newLabels;
	  }
  }
}
