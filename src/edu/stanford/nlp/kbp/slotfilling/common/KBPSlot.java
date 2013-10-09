package edu.stanford.nlp.kbp.slotfilling.common;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.slotfilling.MentionCompatibility;
import edu.stanford.nlp.time.SUTime.IsoDate;

/* Data structure representing a relation mention  */

public class KBPSlot implements Serializable {

  private static final long serialVersionUID = 1L;

  /** The entity name */
  public String entityName;

  /** The entity id */
  public String entityId;

  /** The entity type */
  public EntityType entityType;
  
  /** The filler value, e.g., "33" */
  public String slotValue;

  /** Name of the relation between entity and slot, e.g., "per:age" */
  public String slotName;

  /** If this is a relation predicted by our model, this field stores the document id from which the relation was extracted */
  public String docid;
  
  /** Name of index where this relation comes from */
  public String indexName;

  /**
   * List of matching token spans for the slotValue in the current sentence
   * This is reset for every sentence where this slot matches
   */
  public List<Span> matchingSpans;
  /** Did we match the exact full slot value or a shorter alternate version of the value? */
  public List<Boolean> matchingSpanExact;

  /** Tokenized slot value. Used ONLY during slot matching in sentence, in the reader */
  public String [] slotValueTokens;
  /** Tokenized forms of the alternate slot values. Used ONLY during slot matching in sentence, in the reader */
  public List<String []> alternateSlotValues;
  
  /** 
   * Score assigned to this relation by the predictor
   * This score is assigned after conflict resolution so it may not be a probability anymore 
   */
  double score;

  /**
   * Start and end times, represented as spans in which the start and
   * end could have occurred
   */
  public IsoDate t1Slot, t2Slot, t3Slot, t4Slot;

  public KBPSlot(String entityName, String entityId, 
                            String value, String relation) {
    this.entityName = entityName;
    this.entityId = entityId;
    this.slotValue = value;
    this.slotValueTokens = Utils.tokenizeToStrings(this.slotValue);
    this.slotName = relation.intern();
    this.score = -1;
  }
  
  public void setScore(double s) { score = s; }
  public double getScore() { return score; }
  
  public boolean sameSlot(KBPSlot other) {
    if (other != null && other.slotName.equals(slotName) && 
        other.slotValue.equalsIgnoreCase(slotValue)) 
      return true;
    return false;
  }
  
  @Override
  public String toString() {
    return entityName + ", " + slotName + ": " + slotValue;
  }

  public static boolean isDateSlot(String slotName) {
    return slotName.startsWith("per:date_") ||
            slotName.equals("org:founded") ||
            slotName.equals("org:dissolved");
  }

  private static final Set<String> PERSON_NAME_SLOTS = new HashSet<String>(Arrays.asList(new String[]{
      "per:alternate_names", "per:spouse", "per:children", "per:parents", "per:siblings", "per:other_family",
      "org:top_membersSLASHemployees", "org:shareholders", "org:founded_by"
  }));

  public static boolean isPersonNameSlot(String slotName) {
    return PERSON_NAME_SLOTS.contains(slotName);
  }
  
  private static final Set<String> COUNTRY_NAME_SLOTS = new HashSet<String>(Arrays.asList(new String[]{
      "org:country_of_headquarters",
      "org:member_of",
      "org:parents",
      "per:employee_of",
      "per:origin",
      "per:country_of_birth",
      "per:country_of_death",
      "per:countries_of_residence"
  }));
  
  public static boolean isCountryNameSlot(String slotName) {
    return COUNTRY_NAME_SLOTS.contains(slotName);
  }
  public static boolean isPureCountryNameSlot(String slotName) {
    // NATIONALITY is the prefered NE for per:origin
    return COUNTRY_NAME_SLOTS.contains(slotName) && ! slotName.equals("per:origin");
  }
  
  public void addAlternateSlotValues(SlotsToNamedEntities stone, Nationalities nationalities) {
    MentionCompatibility mc = new MentionCompatibility(nationalities, stone);
    List<String> alternates = mc.findAlternateSlotValues(slotValue, 
        isDateSlot(slotName), isPersonNameSlot(slotName), isCountryNameSlot(slotName));
    for(String alt: alternates) addAlternateName(alt);
    if(alternateSlotValues != null){
      Collections.sort(alternateSlotValues, new Comparator<String []>() {

        public int compare(String[] o1, String[] o2) {
          if(o1.length > o2.length) return -1;
          if(o1.length == o2.length) return 0;
          return 1;
        }
      });
    }
  }
  
  private void addAlternateName(String s) {
    if(alternateSlotValues == null) alternateSlotValues = new ArrayList<String []>();
    String [] stringToks = Utils.tokenizeToStrings(s);
    alternateSlotValues.add(stringToks);
    Log.fine("Added alternate slot value \"" + s + "\" for slot \"" + slotValue + "\" for type " + slotName);
  }
}
