package edu.stanford.nlp.kbp.slotfilling.common;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import edu.stanford.nlp.kbp.slotfilling.distantsupervision.KBPReader;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;

/**
 * Stores a datum generated from a single mention
 * This is used in the old local models
 * @author Mihai
 *
 */
public class MinimalDatum {
  /** Id of this entity in the KB */
  private String entityId;
  /** Type of this entity, e.g., PER or ORG */
  private String entityType;
  /** NE label of this slot candidate */
  private String slotType;
  /** The normalized value of this slot */
  private String slotValue;
  /** The actual datum used in classification */
  private Datum<String, String> datum;
  
  public MinimalDatum(
      String entityId,
      String entityType,
      String slotType,
      String slotValue,
      Datum<String, String> datum) {
    this.entityId = entityId;
    this.entityType = entityType;
    this.slotType = slotType;
    this.slotValue = slotValue;
    this.datum = datum;
  }
  
  public String entityId() { return entityId; }
  public String entityType() { return entityType; }
  public String slotType() { return slotType; }
  public String slotValue() { return slotValue; }
  public Datum<String, String> datum() { return datum; }
  
  public void saveDatum(PrintStream os) {
    os.print("{" + KBPReader.extractEntityId(entityId()) + "} " + 
        entityType().replaceAll("\\s+", "") + " " + 
        slotType().replaceAll("\\s+", "") + " " + 
        slotValue().replaceAll("\\s+", "_") + " " + 
        datum().label());
    Collection<String> feats = ((BasicDatum<String, String>) datum()).asFeatures();
    for(String feat: feats){
      feat = feat.replaceAll("\\s+", "_");
      os.print(" " + feat);
    }
    os.println();
  }
  
  public static List<MinimalDatum> lineToDatum(String line) {
    // the entity id is stored at the beginning within "{...} "
    int entEnd = line.indexOf("} ");
    if(entEnd < 2) throw new RuntimeException("Invalid datum line: " + line);
    String eid = line.substring(1, entEnd);
    line = line.substring(entEnd + 2);
    
    String [] bits = line.split("\\s+");
    String entType = bits[0];
    String neType = bits[1];
    String slotValue = bits[2];
    String concatenatedLabel = bits[3];
    String [] labels = concatenatedLabel.split("\\|");
    if(labels.length > 1){
      Log.finest("Found concatenated label: " + concatenatedLabel);
    }
    Collection<String> feats = new LinkedList<String>();
    for(int i = 4; i < bits.length; i ++){
      String feat = bits[i];
      feats.add(feat);
    }
    List<MinimalDatum> datums = new ArrayList<MinimalDatum>();
    for(String label: labels){
      datums.add(new MinimalDatum(eid, entType, neType, slotValue, new BasicDatum<String, String>(feats, label)));
    }
    return datums;
  }
}
