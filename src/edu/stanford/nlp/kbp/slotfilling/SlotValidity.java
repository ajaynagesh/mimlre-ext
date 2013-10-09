package edu.stanford.nlp.kbp.slotfilling;

import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.kbp.slotfilling.common.SlotToNamedEntities;
import edu.stanford.nlp.kbp.slotfilling.common.SlotsToNamedEntities;

/**
 * Decides if a slot candidate is valid
 * @author Mihai
 *
 */
public class SlotValidity {
  public static boolean validCandidateForLabel(
      SlotsToNamedEntities slotsToNamedEntities,
      String label, 
      String entityType, 
      String slotNE) {
    if (label.equals(RelationMention.UNRELATED)) return true;
    SlotToNamedEntities slotInfo = slotsToNamedEntities.getSlotInfo(label);
    return validCandidateForLabel(slotInfo, label, entityType, slotNE);
  }
  
  public static boolean validCandidateForLabel(SlotToNamedEntities slotInfo,
      String label, 
      String entityType, 
      String slotNE) {
    if (label.startsWith("org:") && 
        ! entityType.startsWith("ENT:ORG") && 
        ! entityType.startsWith("ORG"))
      return false;
    if (label.startsWith("per:") && 
        ! entityType.startsWith("ENT:PER") &&
        ! entityType.startsWith("PER"))
      return false;

    if (!slotInfo.validNamedEntityLabels().contains(slotNE))
      return false;
    
    /*
    boolean validPOS = false;
    for(String prefix: slotInfo.validPOSPrefixes()) {
      if(slotPOS.startsWith(prefix)){
        validPOS = true;
        break;
      }
    }
    if(! validPOS) return false;
    */
    
    // dates must contain at least a YEAR
    // TODO

    return true;
  }

  public static boolean validCandidate(
      SlotsToNamedEntities slotsToNamedEntities, 
      String slotValue,
      String slotNE, 
      String slotPOS, 
      boolean matchSlotNE) {
    // we only care about NEs here
    if(slotNE == null) return false;
    
    // must have a valid NE
    if(matchSlotNE && ! slotsToNamedEntities.getValidNamedEntityLabels().contains(slotNE))
      return false;
    
    // must have a valid POS
    /*
    boolean validPOS = false;
    for(String prefix: slotsToNamedEntities.getValidPOSPrefixes()) {
      if(slotPOS.startsWith(prefix)){
        validPOS = true;
        break;
      }
    }
    if(! validPOS) return false;
    */
    
    // dates must contain at least a YEAR
    // TODO
    
    return true;
  }
}
