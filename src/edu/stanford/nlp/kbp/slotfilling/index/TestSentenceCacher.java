package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;

import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.kbp.slotfilling.common.Utils;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.TaskXMLParser;
import edu.stanford.nlp.kbp.slotfilling.index.IndexExtractor.ResultSortMode;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;

/**
 * Caches sentences for test queries
 */
public class TestSentenceCacher {
  /** How many sentences to fetch from the index per entity */
  private final int sentencesPerEntity;
  
  /** Fetches sentences from the KBP corpus */
  private final SentenceExtractor kbpIndex;

  /** Pipeline just for parsing */
  private final StanfordCoreNLP parserPipeline;
  
  /** Cache entity sentences here */
  private String cacheDir;
  
  private void parseSentence(CoreMap sentence) {
    Tree tree = sentence.get(TreeAnnotation.class);
    if(tree != null) return; // already parsed
    Annotation miniCorpus = new Annotation("");
    List<CoreMap> sents = new ArrayList<CoreMap>();
    sents.add(sentence);
    miniCorpus.set(SentencesAnnotation.class, sents);
    parserPipeline.annotate(miniCorpus);
    assert(sentence.get(TreeAnnotation.class) != null);
  }
  
  public TestSentenceCacher(Properties props) {
    sentencesPerEntity = Integer.parseInt(props.getProperty("kbp.testsentences.per.entity"));
    ResultSortMode sortMode = ResultSortMode.valueOf(props.getProperty(Props.TEST_RESULT_SORT_MODE_PROPERTY, ResultSortMode.NONE.toString()));
    try {
      kbpIndex = IndexExtractor.createIndexExtractors(sortMode, props);
    } catch(IOException e){
      Log.severe("Cannot initialize index from " + props.getProperty(IndexExtractor.DIRECTORY_PROPERTY));
      throw new RuntimeException(e);
    }
    props.setProperty("annotators", "parse");
    parserPipeline = new StanfordCoreNLP(props, false);
    cacheDir = props.getProperty("kbp.testcache");
    assert(cacheDir != null);
  }
  
  public void cache(String path) throws Exception {
    List<KBPEntity> entities = TaskXMLParser.parseQueryFile(path);
    Log.info("Found " + entities.size() + " entities.");
    
    int count = 0;
    for(KBPEntity entity: entities){ // don't use sortedEntities. this way we get a random distribution of examples
      File cacheFile = new File(SentenceCacher.makeCacheFileName(cacheDir, entity.name, entity.type, "cache"));
      if(cacheFile.exists()){
        Log.fine("Skipping entity already cached: " + entity);
        continue;
      }
      
      Log.fine("Searching for entity: " + entity);
      List<CoreMap> sentences = kbpIndex.findRelevantSentences(entity.name, entity.type, sentencesPerEntity, null);
      Log.info("Found the following " + sentences.size() + " sentences containing entity " + entity);
      int i = 1;
      for(CoreMap sent: sentences){
        Log.info("ENTITY: " + entity.name + " SENTENCE #" + i + ": " + Utils.sentenceToMinimalString(sent));
        parseSentence(sent);
        i ++;
      }
   
      SentenceCacher.saveSentences(cacheDir, entity, sentences);

      count ++;
      Log.fine("Processed " + count + " out of " + entities.size() + " entities.");
    }
  }
  
  public static void main(String[] args) throws Exception {
    Log.setLevel(Level.FINE);
    Properties props = StringUtils.argsToProperties(args);
    // enable coref during testing!
    props.setProperty("index.pipelinemethod", "FULL");
    TestSentenceCacher reader = new TestSentenceCacher(props);
    String path = props.getProperty("kbp.testqueries");
    assert(path != null);
    reader.cache(path);
  }
}
