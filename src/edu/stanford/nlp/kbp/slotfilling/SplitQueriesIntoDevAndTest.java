package edu.stanford.nlp.kbp.slotfilling;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.Random;

import edu.stanford.nlp.kbp.slotfilling.common.EntityType;
import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.TaskXMLParser;
import edu.stanford.nlp.util.StringUtils;

/**
 * This splits the 2010 query set into development and testing
 */
public class SplitQueriesIntoDevAndTest {
  static final int DEV_PER = 10; // 10% of total
  static final int DEV_ORG = 10; // 10% of total
  
  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    Log.setLevel(Log.stringToLevel(props.getProperty(Props.LOG_LEVEL, "INFO")));
    
    String queryFile = props.getProperty(Props.TEST_QUERIES);
    List<KBPEntity> mentions = TaskXMLParser.parseQueryFile(queryFile);
    System.err.println("Loaded " + mentions.size() + " queries.");
    List<KBPEntity> pers = extract(mentions, EntityType.PERSON);
    pers = randomize(pers);
    System.err.println("Loaded " + pers.size() + " PERSONs.");
    List<KBPEntity> orgs = extract(mentions, EntityType.ORGANIZATION);    
    System.err.println("Loaded " + orgs.size() + " ORGs.");
    orgs = randomize(orgs);
    
    List<KBPEntity> dev = new ArrayList<KBPEntity>();
    List<KBPEntity> test = new ArrayList<KBPEntity>();
    
    for(int i = 0; i < pers.size(); i ++){
      if(i < DEV_PER) dev.add(pers.get(i));
      else test.add(pers.get(i));
    }
    for(int i = 0; i < orgs.size(); i ++){
      if(i < DEV_ORG) dev.add(orgs.get(i));
      else test.add(orgs.get(i));
    }
    
    makeEntityQueryFile(dev, new File("dev.xml"));
    makeEntityQueryFile(test, new File("test.xml"));
  }
  
  public static void makeEntityQueryFile(List<KBPEntity> ents, File f) throws IOException {
    PrintStream os = new PrintStream(new FileOutputStream(f));
    os.println("<?xml version='1.0' encoding='UTF-8'?>");
    os.println("<kbpslotfill>");
    for(KBPEntity ent: ents){
      os.println("  <query id=\"" + ent.queryId + "\">");
      os.println("    <name>" + ent.name + "</name>");
      os.println("    <docid>" + ent.docid + "</docid>");
      os.println("    <enttype>" + ent.type.toString().substring(0, 3) + "</enttype>");
      os.println("    <nodeid>" + ent.id + "</nodeid>");
      if(ent.ignoredSlots != null && ent.ignoredSlots.size() > 0) {
        os.print("    <ignore>");
        boolean first = true;
        for(String ig: ent.ignoredSlots){
          ig = ig.replace("SLASH", "/");
          if(! first) os.print(" ");
          os.print(ig);
          first = false;
        }
        os.println("</ignore>");
      }
      os.println("  </query>");
    }
    os.println("</kbpslotfill>");    
    os.close();
  }
  
  static List<KBPEntity> randomize(List<KBPEntity> l) {
    // randomize the training set
    Random trainRand = new Random(100);
    KBPEntity [] out = new KBPEntity[l.size()];
    int i = 0;
    for(KBPEntity e: l) out[i ++] = e;
    for(int j = out.length - 1; j > 0; j --){
      int randIndex = trainRand.nextInt(j);
      KBPEntity tmp = out[randIndex];
      out[randIndex] = out[j];
      out[j] = tmp;
    }
    return Arrays.asList(out);
  }
  
  static List<KBPEntity> extract(List<KBPEntity> mentions, EntityType type) {
    List<KBPEntity> out = new ArrayList<KBPEntity>();
    for(KBPEntity e: mentions){
      if(e.type == type) out.add(e);
    }
    return out;
  }
}
