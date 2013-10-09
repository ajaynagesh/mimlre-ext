package edu.stanford.nlp.kbp.slotfilling.common;

import java.util.List;

import edu.stanford.nlp.ie.machinereading.structure.ExtractionObject;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.time.SUTime.IsoDate;
import edu.stanford.nlp.util.CoreMap;

public class TemporalRelationMention extends NormalizedRelationMention {
  private static final long serialVersionUID = 1L;

  public IsoDate t1Slot, t2Slot, t3Slot, t4Slot;

  public TemporalRelationMention(String normSlot, String objectId, CoreMap sentence, Span span, String type, String subtype,
      List<ExtractionObject> args, IsoDate t1Slot, IsoDate t2Slot, IsoDate t3Slot, IsoDate t4Slot) {

    super(normSlot, objectId, sentence, span, type, subtype, args, null);
    this.t1Slot = t1Slot;
    this.t2Slot = t2Slot;
    this.t3Slot = t3Slot;
    this.t4Slot = t4Slot;
  }

  public TemporalRelationMention(String normSlot, String objectId, CoreMap sentence, Span span, String type, String subtype,
      List<ExtractionObject> args) {
    this(normSlot, objectId, sentence, span, type, subtype, args, null, null, null, null);
  }

  /*
   * allot slot value to the four slots
   */
  public void addT1Slot(int slotNumber, IsoDate slotValue) throws Exception {
    switch (slotNumber) {
    case 1:
      this.t1Slot = slotValue;
    case 2:
      this.t2Slot = slotValue;
    case 3:
      this.t3Slot = slotValue;
    case 4:
      this.t4Slot = slotValue;
    default:
      throw new Exception("slot number not recognized");
    }
  }

}
