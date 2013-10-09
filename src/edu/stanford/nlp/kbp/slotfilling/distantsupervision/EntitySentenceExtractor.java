package edu.stanford.nlp.kbp.slotfilling.distantsupervision;

import java.io.File;
import java.io.FileInputStream;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.kbp.slotfilling.common.EntityType;
import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.Utils;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.SourceIndexAnnotation;
import edu.stanford.nlp.kbp.slotfilling.index.KBPAnnotationSerializer;
import edu.stanford.nlp.kbp.slotfilling.index.SentenceCacher;
import edu.stanford.nlp.kbp.slotfilling.webqueries.WebSnippetProcessor;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;

/**
 * Retrieves all sentences for a given entity, from all resources we have (indices, web cache)
 */
public abstract class EntitySentenceExtractor {
  protected static final Logger logger = Logger.getLogger(AllCacheSentenceExtractor.class.getName());
  
  protected int indexSentencesPerEntity;
  
  protected int webSentencesPerEntity;
  
  protected EntitySentenceExtractor(int is, int ws) {
    this.indexSentencesPerEntity = is;
    this.webSentencesPerEntity = ws;
  }
  
  public abstract List<CoreMap> findSentences(KBPEntity entity, Set<String> knownSlots, File sourceFile, boolean testMode, Set<String> validDocIds) throws Exception;
  
  private static final int MAX_CACHE_COUNT = 1000;

  protected static void findWebCachedSentences(
      String entityName, 
      EntityType entityType,
      String webCacheDir,
      int websentencesPerEntity,
      boolean testMode, 
      List<CoreMap> sentences) {
    String pathSuffix = "train" + File.separator + "current";
    if(testMode) pathSuffix = "test" + File.separator + "current";
    boolean useGenericSerialization = WebSnippetProcessor.WebSnippetCacher.USE_GENERIC_SERIALIZATION;
    String fn = SentenceCacher.makeCacheFileName(
        webCacheDir + File.separator + pathSuffix,
        entityName, entityType, (useGenericSerialization ? "cache" : "custom"));
    int count = 0;

    try {
      logger.fine("Reading web cache for entity " + entityName + "...");
      for(int filei = 0; filei < MAX_CACHE_COUNT; filei ++){
        String name = fn + "." + filei;
        File file = new File(name);
        logger.fine("File is |" + file + "|");
        if(! file.exists()) break;

        List<CoreMap> sents = null;
        if(useGenericSerialization) {
          sents = IOUtils.readObjectFromFile(file);
        } else {
          KBPAnnotationSerializer cas = new KBPAnnotationSerializer(true, true);
          FileInputStream is = new FileInputStream(file);
          Annotation corpus = cas.load(is);
          sents = corpus.get(SentencesAnnotation.class);
          assert(sents != null);
        }
        assert(sents != null);
        for(CoreMap sentence : sents){
          sentence.set(SourceIndexAnnotation.class, Constants.WEBINDEX_NAME);
        }
        if(count + sents.size() > websentencesPerEntity){
          sentences.addAll(sents.subList(0, websentencesPerEntity - count));
          count = websentencesPerEntity;
          break;
        }
        sentences.addAll(sents);
        count += sents.size();
        if(count == websentencesPerEntity) break;

        for(CoreMap s: sents){
          logger.fine("WEB SENT for " + entityType + ":" + entityName + ": " + Utils.sentenceToMinimalString(s));
        }
      }
    } catch(Exception e) {
      logger.severe("ERROR: cannot load sentences from web cache " + fn);
      e.printStackTrace();
    }

    logger.info("Read " + count + " sentences from web cache for entity " + entityName);
  }
}
