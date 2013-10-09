package edu.stanford.nlp.kbp.slotfilling.distantsupervision;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.KBPSlot;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.util.StringUtils;

/**
 * Dumps the KB in a tabular format
 */
public class KBPReaderDump {
  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    Log.setLevel(Log.stringToLevel(props.getProperty(Props.LOG_LEVEL)));
    String kbPath = props.getProperty(Props.INPUT_KB);
    assert(kbPath != null);
    KBPReader reader = new KBPReader(props, false, false, false);
    Map<KBPEntity, List<KBPSlot>> entitySlotValues = reader.loadEntitiesAndSlots(kbPath);
    PrintStream os = new PrintStream(new FileOutputStream("kb.tab"));
    for(KBPEntity ent: entitySlotValues.keySet()){
      List<KBPSlot> slots = entitySlotValues.get(ent);
      for(KBPSlot slot: slots) {
        os.println(slot.entityId + "\t" +
            slot.entityName + "\t" +
            slot.entityType + "\t" +
            slot.slotName + "\t" + 
            slot.slotValue);
      }
    }
    os.close();
  }
}
