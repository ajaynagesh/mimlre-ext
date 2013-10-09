package edu.stanford.nlp.kbp.slotfilling.index;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.kbp.slotfilling.common.EntityType;
import edu.stanford.nlp.kbp.slotfilling.common.Utils;
import edu.stanford.nlp.util.CoreMap;

/**
 * Combines the results of any two SentenceExtractors
 */
public class CombinationExtractor implements SentenceExtractor {
  SentenceExtractor[] subextractors;
  public CombinationExtractor(SentenceExtractor ... subextractors) {
    this.subextractors = subextractors;
  }

  /**
   * This implementation keeps getting more results from the
   * subextractors until it has the number of results needed
   */
  public List<CoreMap> findRelevantSentences(String entityName, 
                                             EntityType entityType, int n, Set<String> validDocIds) {
    return findRelevantSentences(entityName, entityType, null, n, validDocIds);
  }
  
  public List<CoreMap> findRelevantSentences(String entityName, 
                                             EntityType entityType, 
                                             Set<String> slotKeywords, int n, Set<String> validDocIds) {
    HashSet<String> uniqueSentences = new HashSet<String>();

    List<CoreMap> results = new ArrayList<CoreMap>();
    for (int i = 0; i < subextractors.length && results.size() < n; ++i) {
      int stillNeeded = n - results.size();
      List<CoreMap> newResults = 
        subextractors[i].findRelevantSentences(entityName, entityType, 
                                               slotKeywords, stillNeeded, validDocIds);
      for (CoreMap result : newResults) {
        // Here we filter sentences so that we only return unique
        // sentences.  Note that if we filter sentences here, that
        // means we may throw away sentences until we don't have n
        // sentences any more.  TODO: we could fix that (most of the
        // time) just by asking for more sentences, but that would be
        // more expensive.  Another solution would be to pass around
        // the set of already-known sentences.  That's not a very
        // happy solution either, though.
        String sentence = Utils.sentenceToString(result, true, false, false,
                                                 false, false, false, false);
        if (!uniqueSentences.contains(sentence)) {
          uniqueSentences.add(sentence);
          results.add(result);
        }

        // note that you don't want to use sublist because a sublist
        // isn't serializable (why should that be?)
        if (results.size() == n) {
          break;
        }
      }
    }
    return results;
  }

  public String timingInformation() {
    StringBuilder timing = new StringBuilder();
    for (int i = 0; i < subextractors.length; ++i) {
      timing.append("Subextractor " + i + ":\n");
      SentenceExtractor extractor = subextractors[i];
      timing.append(extractor.timingInformation());
    }
    return timing.toString();
  }

}
