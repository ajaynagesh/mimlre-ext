package edu.stanford.nlp.kbp.slotfilling.distantsupervision;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.kbp.slotfilling.index.IndexExtractor;
import edu.stanford.nlp.kbp.slotfilling.index.KBPAnnotationSerializer;
import edu.stanford.nlp.kbp.slotfilling.index.SentenceExtractor;
import edu.stanford.nlp.kbp.slotfilling.index.IndexExtractor.ResultSortMode;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationSerializer;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Timing;

public class IndexAndWebCacheSentenceExtractor extends EntitySentenceExtractor {
  /** Combined index for all indices specified in props */
  private SentenceExtractor kbpIndex;

  /** Sentences retrieved from the index are cached here */
  private String indexCacheDir;

  /** Web snippets are cached here; if null do NOT use web */
  private String webCacheDir;

  /** Use known slots in the IR query? */
  boolean useKnownSlots;

  /** Scale query size by this much */
  private double scaleFactor;

  int scaledIndexSentencesPerEntity;
  int scaledWebSentencesPerEntity;

  enum CombineSentencesMode {
    SORT_SEPARATELY, SORT_TOGETHER, NO_SORTING;
  }

  /**
   * how to sort sentences before pruning, either together or separately.
   * default = separately
   */
  CombineSentencesMode combineMode = CombineSentencesMode.NO_SORTING;

  // RelationLM sortingLM = null;

  public IndexAndWebCacheSentenceExtractor(int indexSentencesPerEntity, int webSentencesPerEntity, String indexCacheDir, String webCacheDir, ResultSortMode sortMode,
      Properties props) {
    super(indexSentencesPerEntity, webSentencesPerEntity);
    this.webCacheDir = webCacheDir;
    this.indexCacheDir = indexCacheDir;
    this.useKnownSlots = Boolean.valueOf(props.getProperty(Props.TRAIN_USEKNOWNSLOTS, "true"));

    String scale = props.getProperty("index.indexandweb.extraresults.factor");
    scaleFactor = ((scale != null) ? Double.valueOf(scale) : 1.0);
    scaledIndexSentencesPerEntity = (int) (indexSentencesPerEntity * scaleFactor);
    scaledWebSentencesPerEntity = (int) (webSentencesPerEntity * scaleFactor);

    logger.severe("IndexAndWebCacheSentenceExtractor scaled sentences by " + scaleFactor + "; total index/web to extract " + scaledIndexSentencesPerEntity + "/"
        + scaledWebSentencesPerEntity);

    String combineMode = props.getProperty("index.combinemode");
    if (combineMode != null) {
      this.combineMode = CombineSentencesMode.valueOf(combineMode);
      if(this.combineMode != CombineSentencesMode.NO_SORTING) {
	throw new RuntimeException("ERROR: sorting mode " + combineMode + " not supported in this version!");
      }
    }
    logger.severe("Final sort method at IndexAndWebCacheSentenceExtractor: " + this.combineMode);

    /* // LM not supported
    String lmFile = props.getProperty("index.lm");
    if (lmFile != null) {
      logger.severe("Loading language model " + lmFile);
      Timing lmTime = new Timing();
      Timing.startDoing("Loading language model");
      sortingLM = RelationLM.loadSerialized(lmFile);
      lmTime.done();
    }
    */

    try {
      kbpIndex = IndexExtractor.createIndexExtractors(sortMode, props);
    } catch (IOException e) {
      logger.severe("Cannot initialize index from " + props.getProperty(IndexExtractor.DIRECTORY_PROPERTY));
      throw new RuntimeException(e);
    }

    logger.severe("IndexAndWebCacheSentenceExtractor parameters: indexSentencesPerEntity = " + indexSentencesPerEntity + ", webSentencesPerEntity = " + webSentencesPerEntity
        + ", indexCacheDir = " + indexCacheDir + ", webCacheDir = " + webCacheDir + ", sortMode = " + sortMode);
  }

  private static void saveSentencesToCache(File file, List<CoreMap> sentences) throws IOException {
    PrintStream os = new PrintStream(new FileOutputStream(file));
    AnnotationSerializer ser = new KBPAnnotationSerializer(false, true);
    Annotation corpus = new Annotation("");
    corpus.set(SentencesAnnotation.class, sentences);
    ser.save(corpus, os);
    os.close();
    Log.severe("CACHING: Saved " + sentences.size() + " sentences to cache file " + file.getAbsolutePath());
  }

  private static List<CoreMap> loadCachedSentences(File file, int indexSentencesPerEntity) throws IOException, ClassCastException, ClassNotFoundException {
    AnnotationSerializer ser = new KBPAnnotationSerializer(false, true);
    InputStream is = new FileInputStream(file);
    Annotation corpus = null;
    try {
      corpus = ser.load(is);
    } catch (Exception e) {
      // TODO: handle this correctly. Fix saving/loading of annotations to/from
      // disk.
      Log.severe("ERROR: EXCEPTION CAUGHT in loadCachedSentences for file " + file.getAbsolutePath() + "!!!");
      e.printStackTrace();
      corpus = new Annotation("");
      corpus.set(SentencesAnnotation.class, new ArrayList<CoreMap>());
    }
    is.close();
    List<CoreMap> sents = corpus.get(SentencesAnnotation.class);
    Log.severe("CACHING: Loaded " + (sents != null ? sents.size() : 0) + " sentences from cache file " + file.getAbsolutePath());

    // we may not want all the sentences cached => take the top N (assumes they
    // were ranked in IR)
    if (sents != null && indexSentencesPerEntity < sents.size()) {
      sents = new ArrayList<CoreMap>(sents.subList(0, indexSentencesPerEntity));
    }

    return sents;
  }

  // validocIds should be null if you don't want to restrict the documents from
  // which the information should be extracted
  @Override
  public List<CoreMap> findSentences(KBPEntity entity, Set<String> knownSlots, File sourceFile, boolean testMode, Set<String> validDocIds) throws Exception {
    List<CoreMap> indexSentences = null;

    // check if this entity was already cached. if so, retrieve the cached
    // sentences
    boolean foundInCache = false;
    File myDir = null;
    File myFile = null;
    String sourceName = null;
    if (!testMode) {
      assert (sourceFile != null);
      sourceName = sourceFile.getName();
      if (sourceName.endsWith(".xml")) {
        sourceName = sourceName.substring(0, sourceName.length() - 4);
      }
    }
    if (indexCacheDir != null) {
      String normName = entity.name.trim().toLowerCase().replaceAll("\\s+", "_");
      String type = entity.type.toString();
      myDir = new File(indexCacheDir + File.separator + (testMode ? "test" : "train") + File.separator + // train
                                                                                                         // or
                                                                                                         // test
          (sourceName == null ? "" : sourceName + File.separator) + // indicate
                                                                    // which KB
                                                                    // file this
                                                                    // is from
          "s" + indexSentencesPerEntity + File.separator + // how many sentences
                                                           // per entity
          type + "." + // PER or ORG
          normName.substring(0, Math.min(2, normName.length()))); // actual
                                                                  // entity name
      myFile = new File(myDir.getAbsolutePath() + File.separator + normName + ".cache");

      if (myFile.exists()) {
        indexSentences = loadCachedSentences(myFile, scaledIndexSentencesPerEntity);
        foundInCache = true;
      }
    }

    // caching is disabled or we did not find anything in cache
    if (indexSentences == null) {
      logger.fine("caching is disabled or we did not find anything in cache");
      if (!testMode) {
        // Here is where one can add slot keywords (for training entities only)
        Set<String> slots = null;
        if (useKnownSlots)
          slots = knownSlots;
        indexSentences = kbpIndex.findRelevantSentences(entity.name, entity.type, slots, scaledIndexSentencesPerEntity, validDocIds);
      } else {
        indexSentences = kbpIndex.findRelevantSentences(entity.name, entity.type, scaledIndexSentencesPerEntity, validDocIds);
      }
    }

    // save to cache if these are new sentences
    if (!foundInCache && indexCacheDir != null) {
      // we must have something here, even if it's just an array of size 0
      assert (indexSentences != null);
      myDir.mkdirs();
      saveSentencesToCache(myFile, indexSentences);
    }

    List<CoreMap> webSentences = new ArrayList<CoreMap>();
    // should we use the web cache? if yes, add to sentences
    if (webCacheDir != null) {
      findWebCachedSentences(entity.name, entity.type, webCacheDir, scaledWebSentencesPerEntity, testMode, webSentences);
    }

    List<CoreMap> sentences = combineAndSortSentences(entity, indexSentences, webSentences);
    return sentences;
  }

  public List<CoreMap> combineAndSortSentences(KBPEntity entity, List<CoreMap> indexSentences, List<CoreMap> webSentences) {
    List<CoreMap> sentences = new ArrayList<CoreMap>();
    int maxIndex = Math.min(indexSentencesPerEntity, indexSentences.size());
    int maxWeb = Math.min(webSentencesPerEntity, webSentences.size());
    int maxTotal = Math.min(indexSentencesPerEntity + webSentencesPerEntity, indexSentences.size() + webSentences.size());

    if (/* sortingLM == null || */ combineMode == CombineSentencesMode.NO_SORTING) {
      sentences.addAll(indexSentences.subList(0, maxIndex));
      sentences.addAll(webSentences.subList(0, maxWeb));
    } else {
      throw new RuntimeException("ERROR: unknown sorting mode " + combineMode + " at runtime!");
      /*
      String relation;
      switch (entity.type) {
      case PERSON:
        relation = "per";
        break;
      case ORGANIZATION:
        relation = "org";
        break;
      default:
        throw new IllegalArgumentException("Unhandled entity type " + entity.type);
      }
      switch (combineMode) {
      case SORT_SEPARATELY:
        sortingLM.sort(relation, indexSentences);
        sortingLM.sort(relation, webSentences);
        sentences.addAll(indexSentences.subList(0, maxIndex));
        sentences.addAll(webSentences.subList(0, maxWeb));
        break;
      case SORT_TOGETHER:
        System.out.println(entity.type + " " + entity.name + " " + entity.id + " " + entity.docid + " " + entity.queryId);
        System.out.println("Relation: " + relation);
        sentences.addAll(indexSentences);
        sentences.addAll(webSentences);
        System.out.println("Num sentences: " + sentences.size() + " (max " + maxTotal + ")");
        if (sortingLM.sort(relation, sentences)) {
          sentences = new ArrayList<CoreMap>(sentences.subList(0, maxTotal));
        } else {
          sentences.clear();
          sentences.addAll(indexSentences.subList(0, maxIndex));
          sentences.addAll(webSentences.subList(0, maxWeb));
        }
        break;
      default:
        throw new AssertionError("Unknown sentence combination method");
      }
      */
    }
    return sentences;
  }
}
