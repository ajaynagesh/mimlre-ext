package edu.stanford.nlp.kbp.slotfilling.common;

import edu.stanford.nlp.ling.Datum;

/**
 * This class stores one relation datum and the corresponding RelationMention object from which the datum was generated.
 *
 * @author Mihai
 */
public class DatumAndMention {

  private final Datum<String, String> datum;
  private final NormalizedRelationMention mention;

  public DatumAndMention(Datum<String, String> datum) {
    this(datum, null);
  }

  public DatumAndMention(Datum<String, String> datum, NormalizedRelationMention rel) {
    this.datum = datum;
    this.mention = rel;
  }

  public Datum<String, String> datum() { return datum; }
  public NormalizedRelationMention mention() { return mention; }  
}
