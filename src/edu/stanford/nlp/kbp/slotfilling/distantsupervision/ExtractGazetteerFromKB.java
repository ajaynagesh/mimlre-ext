package edu.stanford.nlp.kbp.slotfilling.distantsupervision;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.KBPSlot;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.util.StringUtils;

/**
 * Builds a gazetteer to be used by regexner from the infoboxes in the KB
 */
public class ExtractGazetteerFromKB {
  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    Log.setLevel(Log.stringToLevel(props.getProperty(Props.LOG_LEVEL)));
    
    // load the regexner map
    BufferedReader is = new BufferedReader(new FileReader(props.getProperty(Props.REGEX_MAP)));
    Set<String> knownTitles = new HashSet<String>();
    for(String line; (line = is.readLine()) != null; ) {
      String [] bits = line.split("\t");
      assert(bits.length >= 2);
      if(bits[1].equals("TITLE")){
        knownTitles.add(bits[0].toLowerCase());
      }
    }
    is.close();
    System.err.println("Loaded " + knownTitles.size() + " known titles."); 
    
    // load KB
    String kbPath = props.getProperty(Props.INPUT_KB);
    assert(kbPath != null);
    KBPReader reader = new KBPReader(props, false, false, false);
    Map<KBPEntity, List<KBPSlot>> entitySlotValues = reader.loadEntitiesAndSlots(kbPath);
    
    // extract titles
    Set<String> newTitles = new HashSet<String>();
    for(KBPEntity ent: entitySlotValues.keySet()){
      List<KBPSlot> slots = entitySlotValues.get(ent);
      for(KBPSlot slot: slots) {
        // TITLE
        if(slot.slotName.equals("per:title")){
          newTitles.add(slot.slotValue.toLowerCase());
        }
      }
    }
    System.err.println("Found " + newTitles.size() + " new titles.");
    
    // save new titles
    int count = 0;
    for(String title: newTitles) {
      if(! knownTitles.contains(title)) {
        System.out.println(title + "\tTITLE");
        count ++;
      }
    }
    System.err.println("Saved " + count + " titles.");
  }
}
