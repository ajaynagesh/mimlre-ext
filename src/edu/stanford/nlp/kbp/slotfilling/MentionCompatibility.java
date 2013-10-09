package edu.stanford.nlp.kbp.slotfilling;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Nationalities;
import edu.stanford.nlp.kbp.slotfilling.common.SlotsToNamedEntities;
import edu.stanford.nlp.kbp.slotfilling.common.Utils;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.KBPReader;
import edu.stanford.nlp.ling.CoreLabel;

/**
 * Decides if two mentions are compatible, i.e., they contain the same entity and same slot value, and can be clustered
 * Note: two mentions with different types can be compatible!
 * @author Mihai
 *
 */
public class MentionCompatibility {
  private final Nationalities nationalities;
  
  private final SlotsToNamedEntities slotsToNamedEntities;
  
  public MentionCompatibility(Nationalities n, SlotsToNamedEntities sn) {
    this.nationalities = n;
    this.slotsToNamedEntities = sn;
  }
  
  /**
   * Decides if two mentions in the wild can be clustered
   * This is used by ConflictResolutionForMentionModel.clusterSlots()
   * This must replicate the behavior
   * implemented in addAlternateSlotValues() and
   * KBPReader.slotMatchesExactOrAlternates()
   * 
   * @param r1
   * @param r2
   */
  public boolean mergeable(RelationMention r1, RelationMention r2) {
    String id1 = KBPReader.extractEntityId(r1.getArg(0).getObjectId());
    String id2 = KBPReader.extractEntityId(r2.getArg(0).getObjectId());
    if (id1.length() == 0)
      throw new RuntimeException("Found entity mention with invalid id: " + r1.getArg(0));
    if (id2.length() == 0)
      throw new RuntimeException("Found entity mention with invalid id: " + r2.getArg(0));
    
    if (id1.equals(id2)) { 
      String s1 = r1.getArg(1).getExtentString();
      String ne1 = r1.getArg(1).getType();
      String s2 = r2.getArg(1).getExtentString();
      String ne2 = r2.getArg(1).getType();
      
      if (s1.equalsIgnoreCase(s2)){
        return true; // everything matches perfectly
      }

      if (compatibleSlots(s1, ne1, s2, ne2) || 
          compatibleSlots(s2, ne2, s1, ne1)) {
        Log.fine("Found compatible relation mentions with different slots: " + r1 + " " + r2);
        return true;
      }
    }
    return false;
  }
  
  private boolean compatibleSlots(String srcSlot, String srcNE, String dstSlot, String dstNE) {
    // Due to their scoring model, it is better if no approximate matches are allowed during testing
    return false;
    //return compatibleSlotsMild(srcSlot, srcNE, dstSlot, dstNE);
    //return compatibleSlotsAggressive(srcSlot, srcNE, dstSlot, dstNE);
  }

  /**
   * Verifies if two slots that might be different strings are actually semantically compatible 
   * See also KBPReader.slotMatchesExactOrAlternates()
   */
  @SuppressWarnings("unused")
  private boolean compatibleSlotsAggressive(String srcSlot, String srcNE, String dstSlot, String dstNE) {
    boolean isDate = slotsToNamedEntities.isNamedEntityForDateSlot(srcNE);
    boolean isPerson = slotsToNamedEntities.isNamedEntityForPersonSlot(srcNE);
    boolean isCountry = slotsToNamedEntities.isNamedEntityForCountrySlot(srcNE);
    
    List<String> srcAlternates = findAlternateSlotValues(srcSlot, isDate, isPerson, isCountry);
    for(String alt: srcAlternates) {
      if(dstSlot.equalsIgnoreCase(alt)){
        return true;
      }
    }
    
    return false;
  }

  @SuppressWarnings("unused")
  private boolean compatibleSlotsMild(String srcSlot, String srcNE, String dstSlot, String dstNE) {
    boolean isDate = slotsToNamedEntities.isNamedEntityForDateSlot(srcNE);
    if (isDate && 
        srcSlot.length() < dstSlot.length() && 
        YEAR.matcher(srcSlot).matches() && 
        dstSlot.indexOf(srcSlot) >= 0) {
      return true;
    }
    return false;
  }
  
  public static final Pattern YEAR = Pattern.compile("[12]\\d\\d\\d");

  private static final Set<String> PERSON_PREFIXES = new HashSet<String>(Arrays.asList(new String[]{
      "mr", "mr.", "ms", "ms.", "mrs", "mrs.", "miss", "mister", "sir", "dr", "dr."
  }));
  private static final Set<String> PERSON_SUFFIXES = new HashSet<String>(Arrays.asList(new String[]{
      "jr", "jr.", "sr", "sr.", "i", "ii", "iii", "iv"
  }));
  
  public static List<String> findPersonAlternateNames(String fullName) {
    List<String> alternates = new ArrayList<String>();
    
    //
    // matching first name last name is fine => remove middle name
    //
    List<CoreLabel> tokens = Utils.tokenize(fullName);
    int start = 0, end = tokens.size() - 1;
    while(start < end){
      if(PERSON_PREFIXES.contains(tokens.get(start).word().toLowerCase())) start ++;
      else break;
    }
    while(end > start){
      if(PERSON_SUFFIXES.contains(tokens.get(end).word().toLowerCase())) end --;
      else break;
    }
    if(start < end - 1){
      String firstlast = tokens.get(start).word() + " " + tokens.get(end).word();
      alternates.add(firstlast);
    }
    
    return alternates;
  }
  
  /**
   * Finds alternate values for the given slot
   * This includes: keeping just the year for dates; removing titles and middle names for person names
   */
  public List<String> findAlternateSlotValues(String slotValue, 
      boolean isDateSlot, 
      boolean isPersonSlot, 
      boolean isCountrySlot) {
    List<String> alternates = new ArrayList<String>();
    
    if(isDateSlot){
      // matching just the year is fine
      Matcher m = YEAR.matcher(slotValue);
      if(m.find()){
        String year = m.group();
        if(year.length() < slotValue.length()){
          alternates.add(year);
        }
      }
    } 
    
    if(isPersonSlot){
      List<String> alternatePersonNames = findPersonAlternateNames(slotValue);
      alternates.addAll(alternatePersonNames);
    } 
    
    if(Constants.COUNTRY_EQ_NATIONALITY){
      if(isCountrySlot && nationalities != null) {
        String nat = nationalities.countryToNationality(slotValue);
        if(nat != null) alternates.add(nat);
      }
    }

    return alternates;
  }
}
