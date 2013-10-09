package edu.stanford.nlp.kbp.slotfilling.common;

import java.util.List;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.DocIDAnnotation;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.IntPair;

public class KBPAnnotations {
  
  private KBPAnnotations() {} // only static members

  /**
   * The CoreMap key for getting the slot mentions corresponding to a sentence.
   * 
   * This key is typically set on sentence annotations.
   */
  public static class SlotMentionsAnnotation implements CoreAnnotation<List<EntityMention>> {
    public Class<List<EntityMention>> getType() {
      return ErasureUtils.<Class<List<EntityMention>>>uncheckedCast(List.class);
    }
  }

  /**
   * This class indicates which index a particular sentence came from.
   * Should be set on a sentence level.
   */
  public static class SourceIndexAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {
      return String.class;
    }
  }

  /**
   * Stores the Lucene (integer) document ID. You can use this to obtain the
   * full Lucene Document for a sentence by retrieving this document from an
   * IndexSearcher. This should not be confused with {@link DocIDAnnotation}
   * which is generally what you want instead of this annotation.
   * Typically set on the sentence level.
   */
  public static class SourceIndexDocIDAnnotation implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {
      return Integer.class;
    }
  }

  public static class DatetimeAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {
      return String.class;
    }
  }

  /**
   * This annotation indicates the positions where the text has been marked
   * (either entity or slot filler)
   */
  public static class MarkedPositionsAnnotation implements CoreAnnotation<List<IntPair>> {
    public Class<List<IntPair>> getType() {
      return ErasureUtils.<Class<List<IntPair>>>uncheckedCast(List.class);
    }
  }
}

