package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import java.util.TreeSet;

import org.apache.lucene.document.Document;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.SimpleFSDirectory;

import edu.stanford.nlp.kbp.slotfilling.common.AntecedentGenerator;
import edu.stanford.nlp.kbp.slotfilling.common.EntityType;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.kbp.slotfilling.common.StringFinder;
import edu.stanford.nlp.kbp.slotfilling.common.Utils;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;

/**
 * PipelineIndexExtractor
 * 
 * @author John Bauer
 */
public class PipelineIndexExtractor extends IndexExtractor {
  public enum PipelineMethod {
    FULL, SPLIT;
  }

  StanfordCoreNLP pipeline = null;
  StanfordCoreNLP step1Pipeline = null, step2Pipeline = null;

  PipelineMethod pipelineMethod;

  static public final String PIPELINE_METHOD_PROPERTY = "index.pipelinemethod";

  public PipelineIndexExtractor(String indexDir, ResultSortMode sortMode, Properties pipelineProperties) throws IOException {
    this(new SimpleFSDirectory(new File(indexDir)), sortMode, pipelineProperties);
    setSource(indexDir);
    Log.fine("Constructed an PipelineIndexExtractor pointing to " + indexDir);
  }

  public PipelineIndexExtractor(Directory directory, ResultSortMode sortMode, Properties pipelineProperties) throws IOException {
    super(directory, sortMode, pipelineProperties);

    // In some situations (e.g., error analysis), we need just minimal NLP
    // analysis
    boolean minimalAnalysis = false;
    if (pipelineProperties != null && pipelineProperties.containsKey(Props.MINIMAL_ANALYSIS)) {
      minimalAnalysis = Boolean.valueOf(pipelineProperties.getProperty(Props.MINIMAL_ANALYSIS));
    }

    try {
      pipelineMethod = PipelineMethod.valueOf(pipelineProperties.getProperty(PIPELINE_METHOD_PROPERTY, PipelineMethod.FULL.toString()));
    } catch (IllegalArgumentException e) {
      throw new IllegalArgumentException("The field " + PIPELINE_METHOD_PROPERTY + " was set with an illegal method, " + pipelineProperties.getProperty("index.pipelinemethod"), e);
    }
    Properties tempProps = new Properties(pipelineProperties);
    switch (pipelineMethod) {
    case FULL:
      if (!minimalAnalysis) {
        tempProps.setProperty("annotators", tempProps.getProperty("index.fullannotators"));
      } else {
        tempProps.setProperty("annotators", "tokenize, ssplit");
      }
      pipeline = new StanfordCoreNLP(tempProps);
      break;
    case SPLIT:
      tempProps.setProperty("annotators", tempProps.getProperty("index.step1annotators"));
      step1Pipeline = new StanfordCoreNLP(tempProps, false);
      tempProps.setProperty("annotators", tempProps.getProperty("index.step2annotators"));
      step2Pipeline = new StanfordCoreNLP(tempProps, false);
      break;
    default:
      throw new AssertionError("Unknown PipelineMethod: " + pipelineMethod);
    }
  }

  public List<CoreMap> findRelevantSentences(Document doc, String entityName, EntityType entityType, int n) {
    List<Document> docs = new ArrayList<Document>(1);
    docs.add(doc);
    return findRelevantSentencesFromDocuments(docs, entityName, entityType, null, n);
  }

  List<CoreMap> findRelevantSentences(List<Integer> docIds, String entityName, EntityType entityType, Set<String> slotKeywords, int n, Set<String> validDocIds) {

    List<Document> docs = new ArrayList<Document>();
    for (Integer docId : docIds) {
      Document doc = fetchDocument(docId);
      String documentId = doc.get(docIdField());
      if (validDocIds == null || (validDocIds != null && documentId != null && validDocIds.contains(documentId)))
        docs.add(doc);

    }
    return findRelevantSentencesFromDocuments(docs, entityName, entityType, slotKeywords, n);
  }

  List<CoreMap> findRelevantSentencesFromDocuments(List<Document> docs, String entityName, EntityType entityType, Set<String> slotKeywords, int n) {
    List<CoreMap> results;
    switch (pipelineMethod) {
    case FULL:
      // TODO: filter the sentences returned by the "full" method by
      // slotKeywords as well?
      results = findFullRelevantSentences(docs, entityName, entityType, n);
      break;
    case SPLIT:
      results = findSplitRelevantSentences(docs, entityName, entityType, slotKeywords, n);
      break;
    default:
      throw new AssertionError("Unknown PipelineMethod: " + pipelineMethod);
    }
    return sortSentences(results, sortMode, slotKeywords, sentenceTooShort, nerTooFew);
  }

  public Set<Integer> findContainingSentences(List<CoreMap> sentences, StringFinder entityFinder, StringFinder slotKeywordFinder, Set<Integer> goodSentences) {
    // Add any sentence that contains a sequence of tokens matching
    // the entity we care about. Note that this block works with a
    // minimal pipeline (just tokenization and sentence splitting)!
    for (int sentenceIndex = 0; sentenceIndex < sentences.size(); ++sentenceIndex) {
      CoreMap sentence = sentences.get(sentenceIndex);
      List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);

      if ((maxSentenceLength <= 0 || tokens.size() <= maxSentenceLength) && entityFinder.matches(sentence) && (slotKeywordFinder == null || slotKeywordFinder.matches(sentence)))
        goodSentences.add(sentenceIndex);
    }

    return goodSentences;
  }

  public List<CoreMap> findSplitRelevantSentences(List<Document> docs, String entityName, EntityType entityType, Set<String> slotKeywords, int n) {
    StringFinder entityFinder = new StringFinder(entityName);
    StringFinder slotKeywordFinder = null;
    if (slotKeywords != null && slotKeywords.size() > 0) {
      slotKeywordFinder = new StringFinder(slotKeywords);
    }

    List<CoreMap> relevantSentences = new ArrayList<CoreMap>();

    int docCount = 0;
    for (Document doc : docs) {
      Log.finest("Starting doc " + docCount);

      String text = extractText(doc);
      if (text == null || text.equals("")) {
        Log.finest("Doc " + docCount + " was empty; ignoring");
        ++docCount;
        continue;
      }

      Annotation annotatedText = new Annotation(text);
      step1Pipeline.annotate(annotatedText);

      List<CoreMap> sentences = annotatedText.get(SentencesAnnotation.class);

      Set<Integer> goodSentences = new TreeSet<Integer>();
      findContainingSentences(sentences, entityFinder, slotKeywordFinder, goodSentences);

      List<CoreMap> reducedSentences = new ArrayList<CoreMap>();
      for (Integer sentence : goodSentences) {
        reducedSentences.add(sentences.get(sentence));
      }
      annotatedText.set(SentencesAnnotation.class, reducedSentences);
      Log.finest("Step #1 complete for one document: found " + reducedSentences.size() + " relevant sentences.");
      for (CoreMap sent : reducedSentences) {
        Log.finest("Sentence for step #2: " + Utils.sentenceToString(sent, true, false, false, false, false, false, false));
      }

      // fill in the rest of the annotations
      step2Pipeline.annotate(annotatedText);

      // todo: fill in coref antecedents? maybe it's not necessary
      // now that all sentences returned as relevant have the queried
      // entity in them

      Log.finest("Step #2 complete for one document.");
      for (Integer sentence : goodSentences) {
        Log.finest("Adding relevant sentence + " + (relevantSentences.size() + 1));
        relevantSentences.add(makeRelevantSentence(sentences.get(sentence), doc));
        if (relevantSentences.size() == n) {
          Log.finest("Found enough sentences; returning");
          return relevantSentences;
        }
      }
      Log.finest("Done with document " + docCount);
      ++docCount;
    }

    return relevantSentences;
  }

  public List<CoreMap> findFullRelevantSentences(List<Document> docs, String entityName, EntityType entityType, int n) {
    List<CoreMap> relevantSentences = new ArrayList<CoreMap>();

    int docCount = 0;
    for (Document doc : docs) {
      Log.finest("Starting doc " + docCount);

      String text = extractText(doc);
      if (text == null || text.equals("")) {
        Log.finest("Doc " + docCount + " was empty; ignoring");
        ++docCount;
        continue;
      }
      Annotation annotatedText = new Annotation(text);
      pipeline.annotate(annotatedText);

      // sanity check: make sure all sentences have a parse tree
      for (CoreMap sent : annotatedText.get(SentencesAnnotation.class)) {
        Tree t = sent.get(TreeAnnotation.class);
        if (t == null) {
          throw new RuntimeException("ERROR: could not generate a parse tree for sentence: " + Utils.sentenceToString(sent));
        }
      }

      AntecedentGenerator antGen = new AntecedentGenerator(entityName, maxSentenceLength);
      List<CoreMap> sentences = annotatedText.get(SentencesAnnotation.class);
      // this find antecedents for all tokens in annotatedTokens,
      // giving precedence to entityName (if it exists in the
      // corresponding chain)
      Set<Integer> goodSentences = antGen.findAntecedents(annotatedText);

      for (Integer sentence : goodSentences) {
        CoreMap sentenceContext = getSentenceContext(sentences, sentence, doc);
        if (sentenceContext == null)
          continue;
        relevantSentences.add(makeRelevantSentence(sentenceContext, doc));
        if (relevantSentences.size() == n)
          return relevantSentences;
      }

      Log.finest("================================");
      ++docCount;
    }

    return relevantSentences;
  }

  public String timingInformation() {
    StringBuilder time = new StringBuilder();
    time.append(luceneTime());
    time.append("\n");
    switch (pipelineMethod) {
    case FULL:
      time.append(pipeline.timingInformation());
      break;
    case SPLIT:
      time.append("Step 1 pipeline:\n" + step1Pipeline.timingInformation());
      time.append("\n");
      time.append("Step 2 pipeline:\n" + step2Pipeline.timingInformation());
      break;
    default:
      throw new AssertionError("Unknown PipelineMethod: " + pipelineMethod);
    }
    return time.toString();
  }
}
