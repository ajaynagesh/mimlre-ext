package edu.stanford.nlp.kbp.slotfilling;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.Map.Entry;

import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.CollectionFactory;
import edu.stanford.nlp.util.CollectionValuedMap;
import edu.stanford.nlp.util.MapFactory;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.StringUtils;

public class ModelCombination {
  
  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    
    String[] inputFiles = PropertiesUtils.getStringArray(props, Props.MODEL_COMBINATION_INPUT_FILES);
    String runId = props.getProperty(Props.RUN_ID);
    double scoreThreshold = PropertiesUtils.getDouble(props, Props.SLOT_THRESHOLD);
    double weightedScoreBias = PropertiesUtils.getDouble(props, Props.MODEL_COMBINATION_SCORE_BIAS);
    boolean anydoc = PropertiesUtils.getBool(props, Props.ANYDOC, 
                                             Constants.DEFAULT_ANYDOC);
    Map<String, String> queryIdToNerType = new HashMap<String, String>(); 
    
    Map<String, KBPEntity> allQueries = KBPEvaluator.loadQueryFile(props.getProperty(Props.TEST_QUERIES));
    Map<String, Set<String>> ignores = computeIgnores(allQueries);
    System.err.println("Slots to be ignored:");
    for(String id: ignores.keySet()) {
      System.err.println("\t" + id + ": " + ignores.get(id));
    }
    
    //
    // load relations from all input files, grouping results into a { (queryId, slotName) : list of matching QueryResult objects }
    //
    
    // on a personal note, this is probably the nastiest Java I've ever written
    MapFactory<Pair<String, String>, Collection<QueryResult>> hashMapFactory = MapFactory.<Pair<String,String>,Collection<QueryResult>>hashMapFactory();
    CollectionFactory<QueryResult> arrayListFactory = CollectionFactory.<QueryResult>arrayListFactory();
    CollectionValuedMap<Pair<String,String>, QueryResult> relationToQueryResults =
      new CollectionValuedMap<Pair<String,String>, QueryResult>(hashMapFactory, arrayListFactory, false);
    
    for (String inputFile : inputFiles) {
      // this determines whether our inputs were produced with model.combination.enabled or not
      boolean inputsHaveScores = PropertiesUtils.getBool(props, Props.MODEL_COMBINATION_INPUTS_HAVE_SCORES);
      for (QueryResult queryResult :  QueryResult.readSystemOutput(inputFile, inputsHaveScores)) {
        String queryId = queryResult.queryId;
        Pair<String,String> key = new Pair<String,String>(queryId, queryResult.slotName);
        relationToQueryResults.add(key, queryResult);
        
        // extract NER types for each query ID
        if (!queryResult.isNull()) {
          String nerType = queryResult.slotName.split(":", 0)[0];
          if (queryIdToNerType.containsKey(queryId)) {
            // if it's already stored, make sure it's not changing as a sanity check
            String oldNerType = queryIdToNerType.get(queryId);
            assert oldNerType.equals(nerType);
          } else {
            queryIdToNerType.put(queryId, nerType);
          }
        }
      } 
    }
    
    // We are missing these entities
    // TODO: fix this!!
    if(! queryIdToNerType.containsKey("SF525")) queryIdToNerType.put("SF525", "org");
    if(! queryIdToNerType.containsKey("SF514")) queryIdToNerType.put("SF514", "org");
    if(! queryIdToNerType.containsKey("SF513")) queryIdToNerType.put("SF513", "org");
    if(! queryIdToNerType.containsKey("SF509")) queryIdToNerType.put("SF509", "org");
    if(! queryIdToNerType.containsKey("SF535")) queryIdToNerType.put("SF535", "org");
    
    //
    // for each (queryId, slotName), merge outputs according to the type of slotName
    //
    List<QueryResult> mergedResults = new ArrayList<QueryResult>();
    for (Entry<Pair<String,String>, Collection<QueryResult>> entry : relationToQueryResults.entrySet()) {
      Pair<String,String> pair = entry.getKey();
      String queryId = pair.first();
      String slotName = pair.second();
      List<QueryResult> queryResults = new ArrayList<QueryResult>(entry.getValue());
      
      // keep track of the set of supporting documents proposed for each slot value and their counts
      // after voting, we will keep the most frequent document
      // this is a map from slot value to counter of doc ids
      Map<String, Counter<String>> docidsForValue = new HashMap<String, Counter<String>>();
      
      // add (optionally weighted) votes from all the matching QueryResult objects
      Counter<String> counts = new ClassicCounter<String>();
      for (QueryResult queryResult : queryResults) {
        double weight = 1;
        
        if (queryResult.isNull()) {
          weight = PropertiesUtils.getDouble(props, Props.MODEL_COMBINATION_NIL_WEIGHT, 1);
        } else if (PropertiesUtils.getBool(props, Props.MODEL_COMBINATION_WEIGHT_BY_SCORES)) {
          weight = weightedScoreBias + queryResult.score;
        }
        counts.incrementCount(queryResult.slotValue, weight);
        
        Counter<String> docids = docidsForValue.get(queryResult.slotValue);
        if(docids == null){
          docids = new ClassicCounter<String>();
          docidsForValue.put(queryResult.slotValue, docids);
        }
        
        if(queryResult.docId != null)
          docids.incrementCount(queryResult.docId);
      }
      
      // determine set of values to keep
      String slotType = SFScore.slotType(queryId + ":" + slotName);
      assert !slotType.equals("error");
      
      Counters.retainAbove(counts, scoreThreshold);
      Set<String> validValues = counts.keySet();
      
      if (validValues.size() > 1) {
        validValues.remove("NIL");
      }
      
      // keep only the queryResults with matching values
      List<QueryResult> matchingQueryResults = new ArrayList<QueryResult>(); 
      
      // sorted to ensure stable output
      Collections.sort(queryResults, new QueryResult.StandardOrdering());
      
      for (QueryResult queryResult : queryResults) {
        String slotValue = queryResult.slotValue;
        if (validValues.contains(slotValue)) {
          QueryResult mergedResult = new QueryResult(queryId, slotName, runId, /*queryResult.docId*/ findMostCommonDoc(docidsForValue.get(slotValue), slotValue), slotValue);
          mergedResult.score = counts.getCount(slotValue);
          matchingQueryResults.add(mergedResult);
        }
      }
      
      // we have to sort and then pick the first item since there may be more than one QueryResult with the highest score
      // this ensures we get consistent output
      if (slotType.equals("single") && matchingQueryResults.size() > 0) {
        mergedResults.add(matchingQueryResults.get(0));
      } else {
        mergedResults.addAll(matchingQueryResults);
      }
    }
    
    //
    // output merged results and score
    //
    
    String workDir = props.getProperty(Props.WORK_DIR);
    String keyFile = props.getProperty(Props.GOLD_RESPONSES);
    assert(keyFile != null);
    
    String outputFilename = workDir + File.separator + runId + ".combined.output";
    File outputFile = new File(outputFilename);
    PrintStream outputFileStream = new PrintStream(new FileOutputStream(outputFile));
    
    QueryResult.writeSystemOutputs(mergedResults, outputFileStream, queryIdToNerType, ignores);   
    SFScore.score(System.out, outputFilename, keyFile, null, anydoc, null, null); // TODO: change null to queryIds as last param
  }
  
  static Map<String, Set<String>> computeIgnores(Map<String, KBPEntity> allQueries) {
    Map<String, Set<String>> ignores = new HashMap<String, Set<String>>();
    for(String id: allQueries.keySet()) {
      KBPEntity query = allQueries.get(id);
      ignores.put(query.queryId, query.ignoredSlots);
    }
    return ignores;
  }
  
  private static String findMostCommonDoc(Counter<String> docids, String slotValue) {
    assert(docids != null);
    
    // docids may contain zero elements, for NIL slots
    if(docids.size() == 0){
      assert(slotValue.equals("NIL"));
      return null;
    }
    
    List<Pair<String,Double>> sortedDocids = Counters.toDescendingMagnitudeSortedListWithCounts(docids);
    assert(sortedDocids.size() > 0);
    String best = sortedDocids.get(0).first();
    @SuppressWarnings("unused")
    double bestVotes = sortedDocids.get(0).second();
    // System.err.println("For slot value \"" + slotValue + "\", picked doc id " + best + " with " + bestVotes + " from the set: " + docids);
    return best;
  }
}