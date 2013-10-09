package edu.stanford.nlp.kbp.slotfilling.common;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;

/**
 * Stores the set of datums for an entire relation, i.e., a tuple (entity, slot)
 * This is used for the "at least once" model
 * @author Mihai
 *
 */
public class RelationDatum {
  /** Id of this entity in the KB */
  private String entityId;
  /** Type of this entity, e.g., PER or ORG */
  private String entityType;
  /** 
   * NE label(s) for each mention of this slot candidate
   * This is typically unique, but it may be more than one due to inconsistencies of NER 
   */
  private List<String> slotTypes;
  /** The normalized value of this slot */
  private String slotValue;
  /** Set of positive labels, i.e., labels we know apply to this tuple */
  private Set<String> yPos;
  /** Set of negative labels, i.e., labels we know do NOT apply to this tuple */
  private Set<String> yNeg;
  private List<Datum<String, String>> datums;
  /** A unique key for this (entity, slot) tuple */
  private String key;
  
  public String entityId() { return entityId; }
  public String entityType() { return entityType; }
  public List<String> slotTypes() { return slotTypes; }
  public String slotValue() { return slotValue; }
  public Set<String> yPos() { return yPos; }
  public Set<String> yNeg() { return yNeg; }
  public List<Datum<String, String>> datums() { return datums; }
  
  /** Constructs a unique key for this (entity, slot value) tuple */
  public String key() {
    if(key != null) return key;
    assert(entityId != null);
    assert(slotValue != null);
    key = entityId + "+" + slotValue;
    return key;
  }
  
  /**
   * Merge info from the other datum into this
   * @param other
   */
  public void merge(RelationDatum other) {
    assert(key().equals(other.key()));
    slotTypes.addAll(other.slotTypes());
    yPos.addAll(other.yPos());
    yNeg.addAll(other.yNeg());
    datums.addAll(other.datums());
  }
  
  public void addNeg(String l) {
    yNeg.add(l);
  }
  
  public String posAsString() {
    return labelsAsString(yPos);
  }
  
  public String negAsString() {
    return labelsAsString(yNeg);
  }
  
  private static String labelsAsString(Set<String> labels) {
    List<String> sorted = new ArrayList<String>(labels);
    Collections.sort(sorted);
    StringBuffer os = new StringBuffer();
    boolean first = true;
    for(String l: sorted) {
      if(! first) os.append("|");
      os.append(l);
      first = false;
    }
    return os.toString();
  }
  
  public RelationDatum(
      String entityId,
      String entityType,
      String slotType,
      String slotValue,
      Set<String> labels,
      Datum<String, String> datum) {
    this.entityId = entityId;
    this.entityType = entityType;
    this.slotTypes = new ArrayList<String>();
    this.slotTypes.add(slotType);
    this.slotValue = slotValue;
    this.yPos = labels;
    this.yNeg = new HashSet<String>(); // generated later when we have full info on this datum
    this.datums = new ArrayList<Datum<String,String>>();
    this.datums.add(datum);
  }
  
  public static RelationDatum lineToDatum(String line) {
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
    if(labels.length == 1 && labels[0].equals(RelationMention.UNRELATED)) {
      // for negatives we set yPos to the empty set
      labels = new String[0];
    }
    Collection<String> feats = new LinkedList<String>();
    for(int i = 4; i < bits.length; i ++){
      String feat = bits[i];
      feats.add(feat);
    }
    RelationDatum relDatum = new RelationDatum(
        eid, entType, neType, slotValue, new HashSet<String>(Arrays.asList(labels)),
        new BasicDatum<String, String>(feats, "")); // we use yPos instead of the datum label
    return relDatum;
  }
}
