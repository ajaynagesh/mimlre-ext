package edu.stanford.nlp.kbp.slotfilling;

import edu.stanford.nlp.kbp.slotfilling.common.KBPTuple;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;

import java.io.*;
import java.util.*;

/**
 * Caches the tuples for a given KBP query
 */
public class QueryCacher {
  public static List<KBPTuple> load(String cacheDirName, String queryId) {
    try {
      List<KBPTuple> tuples = new ArrayList<KBPTuple>();
      BufferedReader is = new BufferedReader(new FileReader(cacheDirName + File.separator + queryId));
      for(KBPTuple tuple; (tuple = loadTuple(is)) != null; )
        tuples.add(tuple);
      is.close();
      return tuples;
    } catch(Exception e) {
      Log.severe("Could not read cache for query " + queryId);
      return null;
    }
  }

  public static void save(String cacheDirName, String queryId, List<KBPTuple> tuples) throws IOException {
    mkDir(cacheDirName);
    PrintStream os = new PrintStream(new FileOutputStream(cacheDirName + File.separator + queryId));
    for(KBPTuple tuple: tuples) {
      saveTuple(os, tuple);
    }
    os.close();
  }

  private static KBPTuple loadTuple(BufferedReader is) throws IOException {
    String entityId = is.readLine();
    if(entityId == null) return null;
    String entityName = is.readLine();
    String entityType = is.readLine();
    String slotType = is.readLine();
    String slotValue = is.readLine();
    int numberOfNormSlots = Integer.parseInt(is.readLine());
    Set<String> normalizedSlotValues = new HashSet<String>();
    for(int i = 0; i < numberOfNormSlots; i ++)
      normalizedSlotValues.add(is.readLine());
    String indexName = is.readLine();
    String docid = is.readLine();
    if(docid.equals("null")) docid = null;
    int size = Integer.valueOf(is.readLine());
    List<String> goldMentionLabels = new ArrayList<String>();
    List<Datum<String, String>> datums = new ArrayList<Datum<String, String>>();
    for(int i = 0; i < size; i ++) {
      goldMentionLabels.add(is.readLine());
      String [] line = is.readLine().split("\t");
      List<String> features = Arrays.asList(line);
      Datum<String, String> datum = new BasicDatum<String, String>(features);
      datums.add(datum);
    }
    return new KBPTuple(
            entityId,
            entityName,
            entityType,
            slotType,
            slotValue,
            normalizedSlotValues,
            indexName,
            docid,
            goldMentionLabels,
            datums);

  }

  private static void saveTuple(PrintStream os, KBPTuple tuple) {
    os.println(tuple.entityId());
    os.println(tuple.entityName());
    os.println(tuple.entityType());
    os.println(tuple.slotType());
    os.println(tuple.slotValue());
    os.println(tuple.normalizedSlotValues().size());
    for(String ns: tuple.normalizedSlotValues())
      os.println(ns);
    os.println(tuple.indexName());
    os.println(tuple.docid());
    os.println(tuple.size());
    for(int i = 0; i < tuple.size(); i ++) {
      os.println(tuple.goldMentionLabel(i));
      // this works for BasicDatum only!
      if(! (tuple.datum(i) instanceof BasicDatum))
        throw new RuntimeException("ERROR: saveTuple only works with BasicDatums!");
      boolean first = true;
      for(String f: tuple.datum(i).asFeatures()) {
        if(! first) os.print("\t");
        os.print(f);
        first = false;
      }
      os.println();
    }
  }

  private static void mkDir(String cacheDirName) {
    File cacheDir = new File(cacheDirName);
    if(! cacheDir.exists() && ! cacheDir.mkdir())
      throw new RuntimeException("ERROR: cannot create query cache directory: " + cacheDirName);
  }
}
