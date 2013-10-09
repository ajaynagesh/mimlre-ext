/**
 * Finds pairs of slot types that can have the same values
 * This is required for the multi label classifier
 * @author Mihai
 */
package edu.stanford.nlp.kbp.slotfilling.distantsupervision;

import java.io.FileOutputStream;
import java.io.PrintStream;
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

public class FindSlotOverlaps {
  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    Log.setLevel(Log.stringToLevel(props.getProperty(Props.LOG_LEVEL)));
    String kbPath = props.getProperty(Props.INPUT_KB);
    assert(kbPath != null);
    KBPReader reader = new KBPReader(props, false, false, false);
    Map<KBPEntity, List<KBPSlot>> entitySlotValues = reader.loadEntitiesAndSlots(kbPath);
    Set<String> overlaps = new HashSet<String>();
    for(KBPEntity ent: entitySlotValues.keySet()){
      List<KBPSlot> slots = entitySlotValues.get(ent);
      for(int i = 0; i < slots.size(); i ++) {
        KBPSlot r1 = slots.get(i);
        for(int j = i + 1; j < slots.size(); j ++){
          KBPSlot r2 = slots.get(j);
          if(r2.slotName.equals(r1.slotName)) continue;
          if(r1.slotValue.equalsIgnoreCase(r2.slotValue)){
            String v = (r1.slotName.compareTo(r2.slotName) < 0 ? r1.slotName + "\t" + r2.slotName : r2.slotName + "\t" + r1.slotName);
            overlaps.add(v);
          }
        }
      }
    }
    PrintStream os = new PrintStream(new FileOutputStream("overlaps.tab"));
    for(String v: overlaps) os.println(v);
    os.close();
  }
}
