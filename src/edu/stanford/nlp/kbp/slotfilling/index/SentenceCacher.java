package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.logging.Level;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.slotfilling.common.EntityType;
import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.KBPSlot;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.kbp.slotfilling.common.Utils;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.SourceIndexAnnotation;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.KBPDomReader;
import edu.stanford.nlp.kbp.slotfilling.index.IndexExtractor.ResultSortMode;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;

/**
 * Caches sentences to disk NO LONGER USED! This is replaced by
 * LucenePipelineCacher.
 */
public class SentenceCacher {

  /**
   * Parses the KBP knowledge base and extracts &lt;entity, slot value&gt;
   * tuples
   */
  private final KBPDomReader domReader;

  /** Fetches sentences from the KBP corpus */
  private final SentenceExtractor kbpIndex;

  /** Pipeline just for parsing */
  private final StanfordCoreNLP parserPipeline;

  /** How many sentences to fetch from the index per entity */
  private final int sentencesPerEntity;

  /**
   * Histogram which counts how many entities we have for a given number of
   * found sentences
   */
  private final Counter<Integer> entToSentHistogram;

  /** Cache entity sentences here */
  private String cacheDir;

  private void parseSentence(CoreMap sentence) {
    Tree tree = sentence.get(TreeAnnotation.class);
    if (tree != null)
      return; // already parsed
    Annotation miniCorpus = new Annotation("");
    List<CoreMap> sents = new ArrayList<CoreMap>();
    sents.add(sentence);
    miniCorpus.set(SentencesAnnotation.class, sents);
    parserPipeline.annotate(miniCorpus);
    assert (sentence.get(TreeAnnotation.class) != null);
  }

  public SentenceCacher(Properties props) throws IOException {
    domReader = new KBPDomReader(props);
    sentencesPerEntity = Integer.parseInt(props.getProperty("sentences.per.entity", "1000"));
    ResultSortMode sortMode = ResultSortMode.valueOf(props.getProperty(Props.TRAIN_RESULT_SORT_MODE_PROPERTY, ResultSortMode.NONE.toString()));
    try {
      kbpIndex = IndexExtractor.createIndexExtractors(sortMode, props);
    } catch (IOException e) {
      Log.severe("Cannot initialize index from " + props.getProperty(IndexExtractor.DIRECTORY_PROPERTY));
      throw new RuntimeException(e);
    }
    entToSentHistogram = new ClassicCounter<Integer>();
    props.setProperty("annotators", "parse");
    parserPipeline = new StanfordCoreNLP(props, false);
    cacheDir = props.getProperty("index.sentencecache");
    assert (cacheDir != null);
  }

  public void cache(String path) throws Exception {
    //
    // fetch slot values for all entities in the KB
    //
    Map<KBPEntity, List<KBPSlot>> entitySlotValues = domReader.parse(path);
    Set<KBPEntity> allEntities = entitySlotValues.keySet();
    Log.info("Found " + allEntities.size() + " known entities in " + path);
    int tupleCount = 0;
    for (KBPEntity key : allEntities) {
      Log.fine("Found entity: " + key);
      Collection<KBPSlot> slots = entitySlotValues.get(key);
      tupleCount += slots.size();
    }
    Log.info("Found " + tupleCount + " tuples for " + allEntities.size() + " entities in " + path);

    //
    // for each entity:
    // - fetch sentences from the index
    // - extract matches of the entity of interest
    // - parse the sentences that contain the entity of interest
    // - save these sentences to disk
    //
    List<KBPEntity> sortedEntities = new ArrayList<KBPEntity>(allEntities);
    Collections.sort(sortedEntities, new Comparator<KBPEntity>() {
      @Override
      public int compare(KBPEntity o1, KBPEntity o2) {
        return o1.name.compareTo(o2.name);
      }
    });
    int count = 0;
    for (KBPEntity entity : allEntities) { // don't use sortedEntities. this way
                                           // we get a random distribution of
                                           // examples
      File cacheFile = new File(makeCacheFileName(cacheDir, entity.name, entity.type, "cache"));
      if (cacheFile.exists()) {
        Log.fine("Skipping entity already cached: " + entity);
        continue;
      }

      Log.fine("Searching for entity: " + entity);
      Collection<KBPSlot> knownSlots = entitySlotValues.get(entity);

      List<CoreMap> sentences = kbpIndex.findRelevantSentences(entity.name, entity.type, PipelineIndexExtractor.slotKeywords(knownSlots, false), sentencesPerEntity, null);

      // List<CoreMap> sentences = kbpIndex.findRelevantSentences(entity.name,
      // entity.type, sentencesPerEntity);
      assert (knownSlots != null);

      // these are tokens of the entity name
      List<CoreMap> goodSentences = new ArrayList<CoreMap>();
      List<CoreLabel> entityTokens = Utils.tokenize(entity.name);
      // find the sentences that contain this entity
      int sentCount = 0;
      for (CoreMap sentence : sentences) {
        // does this entity exist in this sentence?
        if (Utils.contained(entityTokens, sentence.get(TokensAnnotation.class), true)) {
          Log.fine("Found valid sentence: " + Utils.sentenceToString(sentence, true, false, false, false, false, false, false));
          parseSentence(sentence);
          goodSentences.add(sentence);
          sentCount++;
        }
      }
      Log.fine("Found " + sentCount + " sentences in the index containing entity " + entity);
      entToSentHistogram.incrementCount(sentCount);

      saveSentences(cacheDir, entity, goodSentences);

      count++;
      if (count % 10 == 0)
        Log.fine("Processed " + count + " out of " + sortedEntities.size() + " entities.");
    }

    //
    // report some stats
    //
    Log.info("Distribution of entities by the number of sentences found:");
    List<Integer> sentCounts = new ArrayList<Integer>(entToSentHistogram.keySet());
    Collections.sort(sentCounts);
    for (Integer sentCount : sentCounts) {
      Log.info("Entities with " + sentCount + " sentences: " + entToSentHistogram.getCount(sentCount));
    }
  }

  public static String makeCacheDirName(String cacheDir, String name, EntityType type) {
    assert (name.length() > 0);
    String normName = name.toLowerCase().replaceAll("\\s+", "");
    String fnKey = Utils.entityTypeToString(type).substring(0, 3) + "." + normName.substring(0, Math.min(2, normName.length()));
    return (cacheDir + File.separator + fnKey);
  }

  public static String makeCacheDirNameAndCreate(String cacheDir, String name, EntityType type) {
    String dirName = makeCacheDirName(cacheDir, name, type);
    File dir = new File(dirName);
    if (!dir.exists()) {
      if (!dir.mkdir()) {
        Log.severe("Could not create nonexistent directory " + dir);
      }
    }
    return dirName;
  }

  public static String makeCacheFileName(String cacheDir, String name, EntityType type, String extension) {
    assert (name.length() > 0);
    String normName = name.toLowerCase().replaceAll("\\s+", "");
    String fnKey = Utils.entityTypeToString(type).substring(0, 3) + "." + normName.substring(0, Math.min(2, normName.length()));
    String dir = cacheDir + File.separator + fnKey;
    String fn = dir + File.separator + normName + "." + extension;
    return fn;
  }

  public static void saveSentences(String cacheDir, KBPEntity ent, List<CoreMap> sents) {
    File dir = new File(makeCacheDirName(cacheDir, ent.name, ent.type));
    if (!dir.exists()) {
      if (!dir.mkdir()) {
        Log.severe("Could not create nonexistent directory " + dir);
      }
    }
    String serFn = makeCacheFileName(cacheDir, ent.name, ent.type, "cache");
    String debFn = makeCacheFileName(cacheDir, ent.name, ent.type, "debug");
    try {
      IOUtils.writeObjectToFile(sents, new File(serFn));
      PrintStream os = new PrintStream(new FileOutputStream(debFn));
      for (CoreMap sent : sents) {
        saveSentenceDebug(os, sent);
      }
      os.close();
    } catch (IOException e) {
      System.err.println("ERROR: cannot save sentences for entity " + ent.name);
      e.printStackTrace();
    }
  }

  public static void saveSentenceDebug(PrintStream os, CoreMap sent) {
    // generates 4 lines per sentence: 1) index name, 2) the text, 3) the list
    // of tokens, 4) the parse tree

    // line 1: index name
    String indexName = sent.get(SourceIndexAnnotation.class);
    os.println(indexName);

    // line 2: the text
    List<CoreLabel> tokens = sent.get(TokensAnnotation.class);
    boolean first = true;
    for (CoreLabel token : tokens) {
      if (!first)
        os.print(" ");
      os.print(token.word());
      first = false;
    }
    os.println();

    // line 3: the tokens, including AntecedentAnnotation!
    first = true;
    for (CoreLabel token : tokens) {
      if (!first)
        os.print("\t");
      os.print(token.word());
      os.print(" ");
      os.print(token.lemma());
      os.print(" ");
      os.print(token.tag());
      os.print(" ");
      os.print(token.ner());
      os.print(" ");
      String ant = token.get(CoreAnnotations.AntecedentAnnotation.class);
      if (ant != null)
        os.print(ant.replaceAll("\\s+", "_"));
      else
        os.print("NIL");
      first = false;
    }
    os.println();

    // line 4: the parse tree
    Tree tree = sent.get(TreeAnnotation.class);
    os.println(tree);
  }

  public static void main(String[] args) throws Exception {
    Log.setLevel(Level.FINE);
    Properties props = StringUtils.argsToProperties(args);
    SentenceCacher reader = new SentenceCacher(props);
    String path = props.getProperty("kbp.inputkb");
    reader.cache(path);
  }
}
