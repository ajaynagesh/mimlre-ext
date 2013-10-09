package edu.stanford.nlp.kbp.slotfilling.common;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;


public class SlotsToNamedEntities {
  /** Map from slot name to valid NE labels for this slot */
  private Map<String, SlotToNamedEntities> slots;
  /** All valid NE labels for the task */
  private Set<String> validNamedEntityLabels;
  /** All valid POS tag prefixes for the task */
  private Set<String> validPOSPrefixes;
  /** Inverse index from NE labels to slots that accept them */
  private Map<String, Set<String>> neToSlots;
  
  public SlotsToNamedEntities(String path) throws IOException {
    this(new BufferedReader(new FileReader(path)));
  }
  
  public SlotsToNamedEntities(BufferedReader is) throws IOException {
    // we want the results sorted by slot name, and the number of
    // possible slots will be small, so it shouldn't cost us much to
    // use a TreeMap instead of a HashMap
    slots = new TreeMap<String, SlotToNamedEntities>();
    validNamedEntityLabels = new HashSet<String>();
    validPOSPrefixes = new HashSet<String>();
    neToSlots = new HashMap<String, Set<String>>();
    
    readEntries(is);
    
    Log.severe("Read information for " + slots.keySet().size() + " slots: " + slots.keySet());
  }
  
  public boolean isNamedEntityForDateSlot(String ne) {
    for(String slot: neToSlots.get(ne)){
      if(KBPSlot.isDateSlot(slot)) return true;
    }
    return false;
  }
  
  public boolean isNamedEntityForPersonSlot(String ne) {
    for(String slot: neToSlots.get(ne)){
      if(KBPSlot.isPersonNameSlot(slot)) return true;
    }
    return false;
  } 
  
  public boolean isNamedEntityForCountrySlot(String ne) {
    for(String slot: neToSlots.get(ne)){
      if(KBPSlot.isCountryNameSlot(slot)) return true;
    }
    return false;
  }
  
  public void readEntries(BufferedReader reader) throws IOException {
    String line;
    while ((line = reader.readLine()) != null) {
      line = line.trim();
      if (line.equals("")) continue;
      String[] pieces = line.split("\\s+");
      if (pieces.length != 4) {
        throw new RuntimeException("Unexpected file format: '" + line + "'");
      }
      String name = pieces[0];
      String [] types = pieces[2].split("/");
      String [] tags = pieces[3].split("/");
      Set<String> validNELabels = new HashSet<String>(Arrays.asList(types));
      Set<String> validPOSTags = new HashSet<String>(Arrays.asList(tags));
      this.validNamedEntityLabels.addAll(validNELabels);
      this.validPOSPrefixes.addAll(validPOSTags);
      this.slots.put(name, new SlotToNamedEntities(name,
        SlotType.valueOf(pieces[1].toUpperCase()), 
        validNELabels, validPOSTags));
      
      for(String ne: validNELabels) {
        Set<String> mySlots = neToSlots.get(ne);
        if(mySlots == null){
          mySlots = new HashSet<String>();
          neToSlots.put(ne, mySlots);
        }
        mySlots.add(name);
      }
    }
    reader.close();
  }
  
  public SlotToNamedEntities getSlotInfo(String slotName) { return slots.get(slotName); }
  public boolean isListSlot(KBPSlot slot) { return slots.get(slot.slotName).slotType() == SlotType.LIST; }
  public Set<String> getValidNamedEntityLabels() { return validNamedEntityLabels; }
  public Set<String> getValidPOSPrefixes() { return validPOSPrefixes; }
  public Set<String> keySet() { return slots.keySet(); }
  
  public Map<String, Set<String>> buildNeToSlots() {
    Map<String, Set<String>> neToSlots = new HashMap<String, Set<String>>();
    for(String slot: slots.keySet()) {
      for(String ne: slots.get(slot).validNamedEntityLabels()){
        Set<String> ss = neToSlots.get(ne);
        if(ss == null){
          ss = new HashSet<String>();
          neToSlots.put(ne, ss);
        }
        ss.add(slot);
      }
    }
    return neToSlots;
  }
}
