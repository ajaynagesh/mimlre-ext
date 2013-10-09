package edu.stanford.nlp.kbp.slotfilling.common;

import java.util.Set;

public class SlotToNamedEntities implements Comparable<SlotToNamedEntities> {
  /** Slot name, e.g., "org:city_of_headquarters" */
  private String slotName;
  /** Slot type: list or single */
  private SlotType slotType;
  /** Accepted NE labels for this slot, e.g., LOCATION for org:city_of_headquarters */
  private Set<String> validNamedEntityLabels;
  /** Accepted POS tag prefixes for this slot, e.g., NN for org:city_of_headquarters */
  private Set<String> validPOSPrefixes;

  public SlotToNamedEntities(String name, 
      SlotType slotType, 
      Set<String> validNELabels, 
      Set<String> validPOSPrefixes) {
    this.slotName = name;
    this.slotType = slotType;
    this.validNamedEntityLabels = validNELabels;
    this.validPOSPrefixes = validPOSPrefixes;
  }
  
  public String slotName() { return slotName; }
  public SlotType slotType() { return slotType; }
  public Set<String> validNamedEntityLabels() { return validNamedEntityLabels; }
  public Set<String> validPOSPrefixes() { return validPOSPrefixes; }

  @Override
  public String toString() {
    return slotName + ":" + slotType + ":" + validNamedEntityLabels + ":" + validPOSPrefixes;
  }

  @Override
  public int hashCode() {
    assert(slotName != null);
    return slotName.hashCode();
  }
  
  @Override
  public boolean equals(Object obj) {
    if (obj instanceof SlotToNamedEntities) {
      SlotToNamedEntities other = (SlotToNamedEntities) obj;
      assert(slotName != null && other.slotName != null);
      if (other.slotName.equals(slotName)) 
        return true;
    }
    return false;
  }

  public int compareTo(SlotToNamedEntities other) {
    return slotName.compareTo(other.slotName);
  }

}