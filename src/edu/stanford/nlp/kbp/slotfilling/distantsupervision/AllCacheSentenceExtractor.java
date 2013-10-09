package edu.stanford.nlp.kbp.slotfilling.distantsupervision;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.slotfilling.common.EntityType;
import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.index.SentenceCacher;
import edu.stanford.nlp.util.CoreMap;

/**
 * Reads sentences from caches generated during KBP 2010 This is obsolete code;
 * new code should use IndexAndWebCacheSentenceExtractor
 */
public class AllCacheSentenceExtractor extends EntitySentenceExtractor {
  /** Cached entity sentences are stored here */
  private String cacheDir;

  /** Web snippets are cached here; if null do NOT use web */
  private String webCacheDir;

  /** Sentences for test queries are cached here */
  private String queryCacheDir;

  public AllCacheSentenceExtractor(int indexSentencesPerEntity, int webSentencesPerEntity, String cacheDir, String webCacheDir, String queryCacheDir) {
    super(indexSentencesPerEntity, webSentencesPerEntity);
    this.cacheDir = cacheDir;
    this.webCacheDir = webCacheDir;
    this.queryCacheDir = queryCacheDir;
  }

  private List<CoreMap> strip(List<CoreMap> big, int max) {
    if (big.size() <= max)
      return big;
    List<CoreMap> small = new ArrayList<CoreMap>();
    for (int i = 0; i < max; i++)
      small.add(big.get(i));
    return small;
  }

  private List<CoreMap> findTestCachedSentences(String entityName, EntityType entityType, int sentencesPerEntity) {
    String fn = SentenceCacher.makeCacheFileName(queryCacheDir, entityName, entityType, "cache");
    File file = new File(fn);
    return findCachedSentences(file, entityName, entityType, sentencesPerEntity);
  }

  private List<CoreMap> findTrainCachedSentences(String entityName, EntityType entityType, int sentencesPerEntity) {
    String fn = SentenceCacher.makeCacheFileName(cacheDir, entityName, entityType, "cache");
    File file = new File(fn);
    return findCachedSentences(file, entityName, entityType, sentencesPerEntity);
  }

  private List<CoreMap> findCachedSentences(File file, String entityName, EntityType entityType, int sentencesPerEntity) {
    try {
      List<CoreMap> sentences = IOUtils.readObjectFromFile(file);
      sentences = strip(sentences, sentencesPerEntity);
      return sentences;
    } catch (Exception e) {
      logger.severe("ERROR: cannot load sentence from cache file " + file.getAbsolutePath());
      e.printStackTrace();
      return new ArrayList<CoreMap>();
    }
  }

  @Override
  public List<CoreMap> findSentences(KBPEntity entity, Set<String> knownSlots, File sourceFile, boolean testMode, Set<String> validDocIDs) throws Exception {

    if (validDocIDs != null) {
      throw new Exception("restricting the documents by using valid doc ids is not implemented in this class");
    }

    // note: this code does not use knownSlots. knownSlots was used when these
    // sentences were cached (see SentenceCacher)
    List<CoreMap> sentences = null;

    if (!testMode) {
      logger.fine("Train mode: fetching sentences for entity " + entity + " from cache...");
      sentences = findTrainCachedSentences(entity.name, entity.type, indexSentencesPerEntity);
    } else {
      logger.fine("Test mode: fetching sentences for entity " + entity + " from cache...");
      sentences = findTestCachedSentences(entity.name, entity.type, indexSentencesPerEntity);
    }

    // should we use the web cache? if yes, add to sentences
    if (webCacheDir != null) {
      List<CoreMap> webSentences = new ArrayList<CoreMap>();
      findWebCachedSentences(entity.name, entity.type, webCacheDir, webSentencesPerEntity, testMode, webSentences);
      sentences.addAll(webSentences);
    }

    return sentences;
  }
}
