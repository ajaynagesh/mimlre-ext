package edu.stanford.nlp.kbp.slotfilling.common;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Data structure representing the gold responses for a KBP slot-filling task. Given a response
 * file in the 2010 format (usually an assessment_results or participant_annotations file), stores
 * the correct responses for each query. Provides access to these responses, along with information
 * about their equivalence classes.
 * 
 * @author jtibs
 */

public class GoldResponses {
  private static int eclassGenerator = 1000000;
  
  private Map<String, Map<String, Set<Integer>>> queriesToEClasses;
  private Map<Integer, Set<String>> eclassesToResponses;
  
  public GoldResponses(String path) {
    queriesToEClasses = new HashMap<String, Map<String, Set<Integer>>>();
    eclassesToResponses = new HashMap<Integer, Set<String>>();
    
    loadResponses(path);
  }
  
  private void loadResponses(String path) {
    try {
      BufferedReader in = new BufferedReader(new FileReader(path));
      int totalEquivClasses = 0;
      
      while (true) {
        String line = in.readLine();
        if (line == null) break;
        String[] fields = line.split("\t");
        if(fields.length < 11) throw new RuntimeException("Invalid line in gold responses: " + line);
        
        String entityID = fields[1];
        String slotName = fields[3];
        
        String response = fields[8];
        int judgment = Integer.parseInt(fields[10]);
        
        // we only consider valid responses
        if (judgment != 1) continue;
        
        int eclass = Integer.parseInt(fields[9]);
        
        // in case there is a mistake in the file and no equivalence class was assigned 
        // to a correct response, we give it a new, unique equivalence class
        if (eclass == 0) eclass = eclassGenerator++;
        
        // update the map from queries (entityID and slotName) to possible equivalence classes
        if (queriesToEClasses.get(entityID) == null)
          queriesToEClasses.put(entityID, new HashMap<String, Set<Integer>>());
        Map<String, Set<Integer>> slotNamesToClasses = queriesToEClasses.get(entityID);
        
        if (slotNamesToClasses.get(slotName) == null)
          slotNamesToClasses.put(slotName, new HashSet<Integer>());
        if(! slotNamesToClasses.get(slotName).contains(eclass)) totalEquivClasses ++;
        slotNamesToClasses.get(slotName).add(eclass);
          
        // update the map from equivalence classes to responses belonging to that class
        if (eclassesToResponses.get(eclass) == null)
          eclassesToResponses.put(eclass, new HashSet<String>());
        eclassesToResponses.get(eclass).add(response);
      }
      System.err.println("Found a total of " + totalEquivClasses + " equivalence classes.");
    } catch (IOException e) {
      System.err.println("Issue loading judgment file");
      throw new RuntimeException(e);
    }
  }
  
  public Set<Integer> getEquivalenceClasses(String entityID, String slotName) {
    if (queriesToEClasses.get(entityID) == null)
      return null;
    return queriesToEClasses.get(entityID).get(slotName);
  }
  
  public Set<String> getMembers(int eclass) {
    return eclassesToResponses.get(eclass);
  }
  
  /**
   * Returns all possible valid responses for the given query, grouped into
   * sets by equivalence class (all responses in the same set belong to the
   * same equivalence class). Returns null if there is no known correct response
   * for this query.
   */
  public List<Set<String>> getResponses(String entityID, String slotName) {
    Set<Integer> eclasses = getEquivalenceClasses(entityID, slotName);
    if (eclasses == null)
      return null;

    List<Set<String>> result = new ArrayList<Set<String>>();
    
    for (int eclass : eclasses)
      result.add(getMembers(eclass));
    
    return result;
  }
}
