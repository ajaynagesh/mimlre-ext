package edu.stanford.nlp.kbp.slotfilling;

//Scorer for TAC KBP 2011 temporal slot-filling task
//author:  Ralph Grishman

//version 0.1
//May 29, 2011
//initial version:  compares two system outputs
//                ignores doc_id's
//                does not handle V tags (slots with no time information)
//                minimal check for valid dates
//                no check for duplicate values for an entry in a 4-tuple

//version 0.2
//June 18, 2011
//by Qi Li (CUNY, liqiearth@gmail.com)
//fixes formulas for recall and precision (adding 'correctSlots')
//added break down scores for each slot type

//version 0.3
//July 14, 2011
//by Qi Li (CUNY, liqiearth@gmail.com)
//fixes:
//1. when system responds "V" for a slot fill, consider it as a tuple <-inf, +inf, -inf, +inf>, 
// and count it as an output instance (rather than ignore it as in previous version).
//2. print out tab-separated scores 

import java.io.*;
import java.util.*;

public class TSFScore {

  // true to print out judgement for each line of response
  static boolean trace = false;

  // true to ignore case in answers
  static boolean nocase = false;

  // mapping from entity_id:slot_name to map from fills to tuples
  // for key
  static Map<String, Map<String, Tuple>> key;
  // for response
  static Map<String, Map<String, Tuple>> response;

  // slots for temporal task (all are list valued)
  static public Set<String> validSlots = new HashSet<String>();
  static {
    validSlots.add("per:spouse");
    validSlots.add("per:title");
    validSlots.add("per:employee_of");
    validSlots.add("per:member_of");
    validSlots.add("per:cities_of_residence");
    validSlots.add("per:stateorprovinces_of_residence");
    validSlots.add("per:countries_of_residence");
    // validSlots.add("per:schools_attended");
    validSlots.add("org:top_members/employees");
  }

  public static void temporalScore(String responseFile, String keyFile, boolean tr, boolean noCase) throws Exception {

    trace = tr;
    nocase = noCase;

    System.out.println("Temporal SF scorer, ver. 0.3\n");

    // ----------- read in key and response ------------

    key = readFile(keyFile, "key");
    response = readFile(responseFile, "response");

    // ------------- score responses ------------

    // tables for different slot types
    Hashtable<String, Float> table_correct = new Hashtable<String, Float>();
    Hashtable<String, Float> table_spurious = new Hashtable<String, Float>();
    Hashtable<String, Float> table_missing = new Hashtable<String, Float>();
    Hashtable<String, Float> table_totalAns = new Hashtable<String, Float>();
    Hashtable<String, Float> table_totalOutput = new Hashtable<String, Float>();

    for (String query : key.keySet()) {
      String slot_type = getSlotTypeFromQry(query);
      if (!table_correct.contains(slot_type)) {
        Float score = 0.0f;
        table_correct.put(slot_type, score);
      }
      if (!table_spurious.contains(slot_type)) {
        Float score = 0.0f;
        table_spurious.put(slot_type, score);
      }
      if (!table_missing.contains(slot_type)) {
        Float score = 0.0f;
        table_missing.put(slot_type, score);
      }
      if (!table_totalAns.contains(slot_type)) {
        Float score = 0.0f;
        table_totalAns.put(slot_type, score);
      }
      if (!table_totalOutput.contains(slot_type)) {
        Float score = 0.0f;
        table_totalOutput.put(slot_type, score);
      }
    }

    float correct = 0;
    float totalOutput = 0; // added by Qi
    float totalAns = 0; // added by Qi

    for (String query : key.keySet()) {
      Map<String, Tuple> keyValues = key.get(query);
      totalAns += keyValues.size();

      String slot_type = getSlotTypeFromQry(query);
      Float old_Ans = table_totalAns.get(slot_type); // added by Qi
      old_Ans += keyValues.size();
      table_totalAns.put(slot_type, old_Ans);
    }

    for (String query : response.keySet()) {
      Map<String, Tuple> responseValues = response.get(query);
      totalOutput += responseValues.size();

      String slot_type = getSlotTypeFromQry(query);
      Float old_output = table_totalOutput.get(slot_type); // added by Qi
      old_output += responseValues.size();
      table_totalOutput.put(slot_type, old_output);
    }

    for (String query : response.keySet()) {

      String slot_type = getSlotTypeFromQry(query); // added by Qi

      if (trace)
        System.out.println("Scoring responses for query:slot " + query);
      Map<String, Tuple> responseValues = response.get(query);
      Map<String, Tuple> keyValues = key.get(query);
      for (String fill : responseValues.keySet()) {

        if (trace)
          System.out.println("    Scoring responses for slot fill " + fill);
        Tuple responseTuple = responseValues.get(fill);
        Tuple keyTuple = (keyValues == null) ? null : keyValues.get(fill);
        if (keyTuple != null) {
          float s = tupleScore(responseTuple, keyTuple);
          if (trace)
            System.out.println("        Fill is correct, temporal score = " + s);
          correct += s;
          System.out.println("response tuple is " + responseTuple + " and key tuple is " + keyTuple);
          Float old_correct = table_correct.get(slot_type); // added by Qi
          old_correct = old_correct + s;
          table_correct.put(slot_type, old_correct);

          keyValues.remove(fill);
        }
      }
    }

    // report score, tab-separated
    System.out.println("\n\nslotType\tPrecision\tRecall\tF1");

    float recall = correct / totalAns;
    float precision = correct / totalOutput;
    float F = (2 * recall * precision) / (recall + precision);

    System.out.println(String.format("overall\t%.3f\t%.3f\t%.3f", precision, recall, F));

    for (String slot_type : table_totalAns.keySet()) {
      float slot_correct = table_correct.get(slot_type);

      Float slot_totalAns = table_totalAns.get(slot_type);
      Float slot_totalOutput = table_totalOutput.get(slot_type);

      float slot_recall = slot_correct / slot_totalAns;
      float slot_precision = slot_correct / slot_totalOutput;
      float slot_F = (2 * slot_recall * slot_precision) / (slot_recall + slot_precision);

      System.out.println(String.format("%s\t%.3f\t%.3f\t%.3f", slot_type, slot_precision, slot_recall, slot_F));
    }
  }

  /**
   * TSFScorer <response file> <key file> [flags] scores response file against
   * key file
   */

  public static void main(String[] args) throws IOException {

    if (args.length < 2 || args.length > 4) {
      System.out.println("SlotScorer must be invoked with 2 to 4 arguments:");
      System.out.println("\t<response file>  <key file> [flag ...]");
      System.out.println("flags:");
      System.out.println("\ttrace  -- print a line with assessment of each system response");
      System.out.println("\tnocase -- ignore case in matching answer string");
      System.exit(1);
    }
    String responseFile = args[0];
    String keyFile = args[1];
    for (int i = 2; i < args.length; i++) {
      String flag = args[i];
      if (flag.equals("trace")) {
        trace = true;
      } else if (flag.equals("nocase")) {
        nocase = true;
      } else {
        System.out.println("Unknown flag: " + flag);
        System.exit(1);
      }
    }

    System.out.println("Temporal SF scorer, ver. 0.3\n");

    // ----------- read in key and response ------------

    key = readFile(keyFile, "key");
    response = readFile(responseFile, "response");

    // ------------- score responses ------------

    // tables for different slot types
    Hashtable<String, Float> table_correct = new Hashtable<String, Float>();
    Hashtable<String, Float> table_spurious = new Hashtable<String, Float>();
    Hashtable<String, Float> table_missing = new Hashtable<String, Float>();
    Hashtable<String, Float> table_totalAns = new Hashtable<String, Float>();
    Hashtable<String, Float> table_totalOutput = new Hashtable<String, Float>();

    for (String query : key.keySet()) {
      String slot_type = getSlotTypeFromQry(query);
      if (!table_correct.contains(slot_type)) {
        Float score = 0.0f;
        table_correct.put(slot_type, score);
      }
      if (!table_spurious.contains(slot_type)) {
        Float score = 0.0f;
        table_spurious.put(slot_type, score);
      }
      if (!table_missing.contains(slot_type)) {
        Float score = 0.0f;
        table_missing.put(slot_type, score);
      }
      if (!table_totalAns.contains(slot_type)) {
        Float score = 0.0f;
        table_totalAns.put(slot_type, score);
      }
      if (!table_totalOutput.contains(slot_type)) {
        Float score = 0.0f;
        table_totalOutput.put(slot_type, score);
      }
    }

    float correct = 0;
    float totalOutput = 0; // added by Qi
    float totalAns = 0; // added by Qi

    for (String query : key.keySet()) {
      Map<String, Tuple> keyValues = key.get(query);
      totalAns += keyValues.size();

      String slot_type = getSlotTypeFromQry(query);
      Float old_Ans = table_totalAns.get(slot_type); // added by Qi
      old_Ans += keyValues.size();
      table_totalAns.put(slot_type, old_Ans);
    }

    for (String query : response.keySet()) {
      Map<String, Tuple> responseValues = response.get(query);
      totalOutput += responseValues.size();

      String slot_type = getSlotTypeFromQry(query);
      Float old_output = table_totalOutput.get(slot_type); // added by Qi
      old_output += responseValues.size();
      table_totalOutput.put(slot_type, old_output);
    }

    for (String query : response.keySet()) {

      String slot_type = getSlotTypeFromQry(query); // added by Qi

      if (trace)
        System.out.println("Scoring responses for query:slot " + query);
      Map<String, Tuple> responseValues = response.get(query);
      Map<String, Tuple> keyValues = key.get(query);
      for (String fill : responseValues.keySet()) {

        if (trace)
          System.out.println("    Scoring responses for slot fill " + fill);
        Tuple responseTuple = responseValues.get(fill);
        Tuple keyTuple = (keyValues == null) ? null : keyValues.get(fill);
        if (keyTuple != null) {
          float s = tupleScore(responseTuple, keyTuple);
          System.out.println("response is " + responseTuple.toString());
          System.out.println("key is " + keyTuple.toString());
          if (trace)
            System.out.println("        Fill is correct, temporal score = " + s);
          correct += s;

          Float old_correct = table_correct.get(slot_type); // added by Qi
          old_correct = old_correct + s;
          table_correct.put(slot_type, old_correct);

          keyValues.remove(fill);
        }
      }
    }

    // report score, tab-separated
    System.out.println("\n\nslotType\tPrecision\tRecall\tF1");

    float recall = correct / totalAns;
    float precision = correct / totalOutput;
    float F = (2 * recall * precision) / (recall + precision);

    System.out.println(String.format("overall\t%.3f\t%.3f\t%.3f", precision, recall, F));

    for (String slot_type : table_totalAns.keySet()) {
      float slot_correct = table_correct.get(slot_type);

      Float slot_totalAns = table_totalAns.get(slot_type);
      Float slot_totalOutput = table_totalOutput.get(slot_type);

      float slot_recall = slot_correct / slot_totalAns;
      float slot_precision = slot_correct / slot_totalOutput;
      float slot_F = (2 * slot_recall * slot_precision) / (slot_recall + slot_precision);

      System.out.println(String.format("%s\t%.3f\t%.3f\t%.3f", slot_type, slot_precision, slot_recall, slot_F));
    }
  }

  /**
   * get slot type of query
   * 
   * @param query
   */
  private static String getSlotTypeFromQry(String query) {
    int idx = query.indexOf(':');
    String slot_type = query.substring(idx + 1, query.length());
    return slot_type;
  }

  /**
   * read a file with temporal slot filling system output, returning a map from
   * entity_id:slot_name to a map from slot_fill to date tuple.
   */

  static Map<String, Map<String, Tuple>> readFile(String fileName, String keyOrResponse) throws IOException {
    BufferedReader reader = null;
    try {
      reader = new BufferedReader(new FileReader(fileName));
    } catch (FileNotFoundException e) {
      System.out.println("Unable to open " + keyOrResponse + " file " + fileName);
      System.exit(1);
    }

    Map<String, Map<String, Tuple>> data = new TreeMap<String, Map<String, Tuple>>();

    String line;
    int lineCount = 0;
    reading: while ((line = reader.readLine()) != null) {
      lineCount++;
      if (line.startsWith("#"))
        continue reading;
      String[] fields = line.split("\t", 7);
      if (fields.length != 7) {
        System.out.println("Invalid line in " + keyOrResponse + " file:");
        System.out.println(line);
        continue reading;
      }
      String entity_id = fields[0];
      String slot_name = fields[1];
      if (!validSlots.contains(slot_name)) {
        System.out.println("Invalid slot name in " + keyOrResponse + " file:");
        System.out.println(line);
        continue reading;
      }
      String Ti = fields[2];
      if (Ti.equals("NIL"))
        continue reading;

      // String run = fields[4];
      // String doc_id = fields[5];
      String fill = fields[6];
      if (nocase)
        fill = fill.toLowerCase();

      String query_id = entity_id + ":" + slot_name; // Qi's comment: query_id
      // is combination of
      // entity_id + slot_type
      Map<String, Tuple> values = data.get(query_id);
      if (values == null) {
        values = new TreeMap<String, Tuple>();
        data.put(query_id, values);
      }

      // system fail to find any temporal information for this fill
      if (Ti.equals("V")) {
        Tuple tuple = values.get(fill);
        if (tuple != null)
          System.err.println("Multiple tuple for " + query_id + "\t " + fill);
        // make a default tuple <inf, inf, inf, inf> for V
        tuple = new Tuple();
        values.put(fill, tuple);
        continue;
      }

      if (!(Ti.equals("T1") || Ti.equals("T2") || Ti.equals("T3") || Ti.equals("T4"))) {
        System.out.println("Invalid T field in " + keyOrResponse + " file:");
        System.out.println(line);
        continue reading;
      }

      String dateString = fields[3];
      int date;
      try {
        date = Integer.parseInt(dateString);
      } catch (NumberFormatException e) {
        System.out.println("Invalid date in " + keyOrResponse + " file:");
        System.out.println(line);
        continue reading;
      }
      if (date < 19000000 || date > 21000000) {
        System.out.println("Invalid date in " + keyOrResponse + " file:");
        System.out.println(line);
        continue reading;
      }

      Tuple tuple = values.get(fill);
      if (tuple == null) {
        tuple = new Tuple();
        values.put(fill, tuple);
      }
      tuple.set(Ti, date);
    }
    System.out.println("Read " + lineCount + " lines from " + keyOrResponse + " file.");
    return data;
  }

  /**
   * temporal score (from section 3.4.3 of 2011 KBP task specification.
   * Currently both vagueness and overcontraining constants are set to 1.
   */

  static float tupleScore(Tuple a, Tuple b) {
    return tScore(a.T1, b.T1) + tScore(a.T2, b.T2) + tScore(a.T3, b.T3) + tScore(a.T4, b.T4);
  }

  static float tScore(long a, long b) {
    return 0.25f * (1.f / (1.f + Math.abs(years(a) - years(b))));
  }

  /**
   * converts an eight digit (yyyymmdd) date to a number of years.
   */

  static float years(long date) {
    float day = date % 100;
    float month = (date / 100) % 100;
    float year = date / 10000;
    return year + month / 12.f + day / 365.f;
  }

  /**
   * a quadruple of dates.
   */
  static class Tuple {

    long T1 = -1000000000;
    long T2 = +1000000000;
    long T3 = -1000000000;
    long T4 = +1000000000;

    void set(String field, int value) {
      if (field.equals("T1"))
        T1 = value;
      else if (field.equals("T2"))
        T2 = value;
      else if (field.equals("T3"))
        T3 = value;
      else if (field.equals("T4"))
        T4 = value;
    }

    @Override
    public String toString() {
      return "{" + T1 + ", " + T2 + ". " + T3 + ", " + T4 + "}";
    }
  }

}