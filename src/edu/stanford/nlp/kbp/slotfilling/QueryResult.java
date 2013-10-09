package edu.stanford.nlp.kbp.slotfilling;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import org.apache.commons.lang.StringUtils;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.util.ArrayUtils;
import edu.stanford.nlp.util.CollectionValuedMap;
import edu.stanford.nlp.util.TwoDimensionalCollectionValuedMap;

/**
 * 
 */
public class QueryResult {

  public String queryId, slotName, runId, docId, slotValue;
  public double score = -1;
  
  public QueryResult(String queryId, String slotName, String runId, String docId, String slotValue) {
    this.queryId = queryId;
    this.slotName = slotName;
    this.runId = runId;
    this.docId = docId;
    this.slotValue = slotValue;
  }
  
  public QueryResult(String line) {
    this(line, false);
  }
  
  public QueryResult(String line, boolean modelCombinationFormat) {
    String[] pieces = StringUtils.splitByWholeSeparator(line, " ", modelCombinationFormat ? 6 : 5);
    queryId = pieces[0];
    slotName = pieces[1];
    runId = pieces[2];
    
    if (pieces.length == 4) {
      docId = null;
      assert pieces[3].equals("NIL");
      slotValue = "NIL";
    } else {
      assert !pieces[3].equals("NIL");
      docId = pieces[3];
      
      if (modelCombinationFormat) {
        score = Double.parseDouble(pieces[4]);
        slotValue = pieces[5].trim();
      } else {
        slotValue = pieces[4].trim();
      }
    }
  }
  
  public boolean isNull() {
    return slotValue.equals("NIL");
  }
  
  @Override
  public String toString() {
    return "QueryResult [docId=" + docId + ", queryId=" + queryId + ", runId="
        + runId + ", slotName=" + slotName + ", slotValue=" + slotValue + ", score=" + score + "]";
  }
  
  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((docId == null) ? 0 : docId.hashCode());
    result = prime * result + ((queryId == null) ? 0 : queryId.hashCode());
    result = prime * result + ((runId == null) ? 0 : runId.hashCode());
    long temp;
    temp = Double.doubleToLongBits(score);
    result = prime * result + (int) (temp ^ (temp >>> 32));
    result = prime * result + ((slotName == null) ? 0 : slotName.hashCode());
    result = prime * result + ((slotValue == null) ? 0 : slotValue.hashCode());
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof QueryResult)) {
      return false;
    }
    QueryResult other = (QueryResult) obj;
    if (docId == null) {
      if (other.docId != null) {
        return false;
      }
    } else if (!docId.equals(other.docId)) {
      return false;
    }
    if (queryId == null) {
      if (other.queryId != null) {
        return false;
      }
    } else if (!queryId.equals(other.queryId)) {
      return false;
    }
    if (runId == null) {
      if (other.runId != null) {
        return false;
      }
    } else if (!runId.equals(other.runId)) {
      return false;
    }
    if (Double.doubleToLongBits(score) != Double.doubleToLongBits(other.score)) {
      return false;
    }
    if (slotName == null) {
      if (other.slotName != null) {
        return false;
      }
    } else if (!slotName.equals(other.slotName)) {
      return false;
    }
    if (slotValue == null) {
      if (other.slotValue != null) {
        return false;
      }
    } else if (!slotValue.equals(other.slotValue)) {
      return false;
    }
    return true;
  }

  public void writeSystemOutput(PrintStream ps, boolean modelCombinationFormat) {
    ps.print(queryId + " " + slotName + " " + runId + " ");
    if (isNull()) {
      ps.println("NIL");
    } else {
      String suffix = docId + " " + slotValue;
      if (modelCombinationFormat) {
        ps.println(score + " " + suffix);
      } else {
        ps.println(suffix);
      }
    }
  }
  
  public void writeSystemOutput(PrintStream ps) {
    writeSystemOutput(ps, false);
  }
  
  //
  // static utility methods and a Comparator
  //
  
  public static List<QueryResult> readSystemOutput(String systemOutputFilename, boolean modelCombinationFormat) {
    List<QueryResult> queryResults = new ArrayList<QueryResult>();
    for (String line : IOUtils.readLines(systemOutputFilename)) {
      queryResults.add(new QueryResult(line, modelCombinationFormat));
    }
    return queryResults;
  }
  
  public static void writeSystemOutputs(Collection<QueryResult> queryResults, PrintStream stream, Map<String, String> queryIdToNerType, Map<String, Set<String>> queryIdToIgnoredSlots) {
    String runId = null; // we will autodetect this
    
    // expand queryResults to include NILs for all remaining slots
    TwoDimensionalCollectionValuedMap<String, String, QueryResult> queryIdToSlotNameToQueryResults = new TwoDimensionalCollectionValuedMap<String, String, QueryResult>();
    for (QueryResult queryResult : queryResults) {
      queryIdToSlotNameToQueryResults.add(queryResult.queryId, queryResult.slotName, queryResult);
      if (runId == null) {
        runId = queryResult.runId;
      } else {
        assert runId.equals(queryResult.runId);
      }
    }
    
    for (Entry<String, CollectionValuedMap<String, QueryResult>> entry : queryIdToSlotNameToQueryResults.entrySet()) {
      String queryId = entry.getKey();
      String nerType = queryIdToNerType.get(queryId);
      List<String> matchingRelations;
      if (nerType == null) {
        throw new RuntimeException("Found NULL NE type for query " + queryId);
        // matchingRelations = SFScore.allSlots;
      } else {
        System.err.println("Using NE type " + nerType + " for query " + queryId);
        matchingRelations = SFScore.relationsForNerType(nerType);
      }
      CollectionValuedMap<String, QueryResult> slotNameToQueryResults = entry.getValue();
      for (String relation : matchingRelations) {
        if (slotNameToQueryResults.get(relation).size() == 0) {
          // make a NIL relation
          QueryResult nilResult = new QueryResult(queryId, relation, runId, null, "NIL");
          slotNameToQueryResults.add(relation, nilResult);
        }
      }
    }
    
    List<QueryResult> expandedQueryResults = new ArrayList<QueryResult>(queryIdToSlotNameToQueryResults.values());
    
    Collections.sort(expandedQueryResults, new QueryResult.StandardOrdering());
    for (QueryResult mergedResult : expandedQueryResults) {
      Set<String> myIgnores = null; 
      assert(queryIdToIgnoredSlots != null);
      if(queryIdToIgnoredSlots != null) {
        myIgnores = queryIdToIgnoredSlots.get(mergedResult.queryId);
      }
      // assert(myIgnores != null);
      if(myIgnores == null || ! myIgnores.contains(mergedResult.slotName)){
        mergedResult.writeSystemOutput(stream);
      } else {
        System.err.println("Skipping <ignore> slot " + mergedResult.slotName + " for query " + mergedResult.queryId);
      }
    }
  }

  public static class StandardOrdering implements Comparator<QueryResult> {

    public int compare(QueryResult o1, QueryResult o2) {
      return ArrayUtils.compareArrays(extractFields(o1), extractFields(o2));
    }

    private String[] extractFields(QueryResult queryResult) {
      String docId = queryResult.docId;
      if (docId == null) {
        docId = "(null)";
      }
      return new String[] { queryResult.queryId, queryResult.slotName, queryResult.slotValue, docId, queryResult.runId };
    }    
  }
  
  public static class ByScore implements Comparator<QueryResult> {

    public int compare(QueryResult o1, QueryResult o2) {
      return -Double.compare(o1.score, o2.score);
    }
  }
}