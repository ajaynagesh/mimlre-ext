package edu.stanford.nlp.kbp.slotfilling.index;

import java.util.List;
import java.util.Set;

import edu.stanford.nlp.kbp.slotfilling.common.EntityType;
import edu.stanford.nlp.util.CoreMap;

/**
 * A class that overrides this interface provides a method for which,
 * given an entity name and type, it returns n sentences pertaining to
 * that entity in the form of CoreMaps.
 */
public interface SentenceExtractor {
  /**
   * the validDocIds should be null if you don't want to restrict the doc ids
   */
  public List<CoreMap> findRelevantSentences(String entityName, EntityType entityType, int n, Set<String> validDocIds);
  
  /**
   * the validDocIds should be null if you don't want to restrict the doc ids
   */
  public List<CoreMap> findRelevantSentences(String entityName, EntityType entityType, Set<String> slotKeywords, int n, Set<String> validDocIds);

  public String timingInformation();
}
