package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import java.util.logging.Level;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.lang.StringEscapeUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;

import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.kbp.slotfilling.common.EntityType;
import edu.stanford.nlp.kbp.slotfilling.common.KBPSlot;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.StringFinder;
import edu.stanford.nlp.kbp.slotfilling.common.Utils;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.DatetimeAnnotation;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.SourceIndexAnnotation;
import edu.stanford.nlp.kbp.slotfilling.index.WikipediaReader.ListRemoval;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.DocIDAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Timing;

/**
 * IndexExtractor
 * 
 * @author John Bauer <br>
 *         Sample command line: <br>
 *         java -mx8g edu.stanford.nlp.kbp.index.IndexExtractor -props
 *         ~/codebase
 *         /javanlp/projects/research/src/edu/stanford/nlp/kbp/kbp.properties
 *         -index.debugquery "u.s. department" -index.debugsentences 1000
 *         -index.sortmode NONE > usdept.txt 2>&1 <br>
 *         Another: <br>
 *         java -mx8g edu.stanford.nlp.kbp.index.IndexExtractor -props
 *         ~/codebase
 *         /javanlp/projects/research/src/edu/stanford/nlp/kbp/kbp.properties
 *         -index.debugquery "china news agency" -index.debugsentences 1000
 *         -index.sortmode NONE -index.kbp
 *         /u/nlp/data/TAC-KBP2010/indices/TAC_2010_KBP_Source_Data > china.txt
 *         2>&1
 */
public abstract class IndexExtractor implements SentenceExtractor {
  public enum ResultSortMode {
    NONE, NER, BUCKETS, TRIGGER;
  }

  /**
   * Accept only sentences up to this length.
   */
  int maxSentenceLength;

  IndexSearcher searcher;
  Set<String> stopWords;

  String source = null;

  public String getSource() {
    return source;
  }

  public void setSource(String source) {
    this.source = source;
  }

  static public final String DIRECTORY_PROPERTY = "index.kbp";
  static public final String DEBUG_SENTENCES_PROPERTY = "index.debugsentences";
  static public final String DEBUG_QUERY_PROPERTY = "index.debugquery";
  static public final String DEBUG_SLOT_PROPERTY = "index.debugslots";
  static public final String NER_TOO_FEW_PROPERTY = "index.nertoofew";
  static public final String SENTENCE_TOO_SHORT_PROPERTY = "index.sentencetooshort";

  static public final String DEFAULT_SENTENCE_TOO_SHORT = "15";
  static public final String DEFAULT_NER_TOO_FEW = "1";
  static public final String MAX_SENT_LEN_PROPERTY = "index.maxsentencelength";
  static public final String DEFAULT_MAX_SENT_LEN = "100";

  static public final String PREVIOUS_CONTEXT_PROPERTY = "index.context.previous";
  static public final String DEFAULT_PREVIOUS_CONTEXT = "0";
  static public final String NEXT_CONTEXT_PROPERTY = "index.context.next";
  static public final String DEFAULT_NEXT_CONTEXT = "0";

  int sentenceTooShort;
  int nerTooFew;

  ResultSortMode sortMode;

  long totalLuceneTime = 0;
  long totalQueryTime = 0;
  long totalFetchTime = 0;

  /**
   * How many sentences before or after the chosen sentence to use when
   * returning results. This behavior is available for a full pipeline extractor
   * or a cached extractor, but not a split pipeline extractor.
   */
  int previousContext, nextContext;

  private boolean isWikipedia = false;

  public void setIsWikipedia(boolean isW) {
    this.isWikipedia = isW;
  }

  public boolean getIsWikipedia() {
    return this.isWikipedia;
  }

  static public SentenceExtractor createIndexExtractors(ResultSortMode sortMode, Properties pipelineProperties) throws IOException {
    String directoryProperty = pipelineProperties.getProperty(DIRECTORY_PROPERTY);
    if (directoryProperty == null) {
      throw new NullPointerException("The provided properties file did not have " + DIRECTORY_PROPERTY + " set");
    }

    String[] directories = directoryProperty.split(";");
    if (directories.length == 1) {
      Log.severe("Creating index extractor on: " + directories[0]);
      return createIndexExtractor(directories[0], sortMode, pipelineProperties);
    } else {
      SentenceExtractor[] extractors = new SentenceExtractor[directories.length];
      for (int i = 0; i < directories.length; ++i) {
        Log.severe("Creating index extractor on: " + directories[i]);
        extractors[i] = createIndexExtractor(directories[i], sortMode, pipelineProperties);
      }
      return new CombinationExtractor(extractors);
    }
  }

  static public SentenceExtractor createIndexExtractor(String indexLabel, ResultSortMode sortMode, Properties pipelineProperties) throws IOException {
    String[] pieces = indexLabel.split(",");
    String indexDir = pieces[pieces.length - 1];
    if (pieces.length > 2)
      throw new IllegalArgumentException("Only expected [<indextype>,]<dir>;" + " got " + indexLabel);
    if (pieces.length == 2 && pieces[0].equals("cached")) {
      return new CachedIndexExtractor(indexDir, sortMode, pipelineProperties);
    }
    PipelineIndexExtractor extractor = new PipelineIndexExtractor(indexDir, sortMode, pipelineProperties);

    // Wikipedia indices need markup removal; so keep track if its Wikipedia
    if (pieces.length == 2 && pieces[0].equals("wiki")) {
      Log.finest(indexDir + " is a wikipedia dir");
      extractor.setIsWikipedia(true);
    }

    return extractor;
  }

  public IndexExtractor(Directory directory, ResultSortMode sortMode, Properties pipelineProperties) throws IOException {
    searcher = new IndexSearcher(directory);
    stopWords = QueryUtils.standardStopWords();

    this.sentenceTooShort = Integer.parseInt(pipelineProperties.getProperty(SENTENCE_TOO_SHORT_PROPERTY, DEFAULT_SENTENCE_TOO_SHORT));
    this.nerTooFew = Integer.parseInt(pipelineProperties.getProperty(NER_TOO_FEW_PROPERTY, DEFAULT_NER_TOO_FEW));
    this.sortMode = sortMode; // ResultSortMode.valueOf(pipelineProperties.getProperty(RESULT_SORT_MODE_PROPERTY,
    // ResultSortMode.NONE.toString()));
    this.maxSentenceLength = Integer.parseInt(pipelineProperties.getProperty(MAX_SENT_LEN_PROPERTY, DEFAULT_MAX_SENT_LEN));

    this.previousContext = Integer.parseInt(pipelineProperties.getProperty(PREVIOUS_CONTEXT_PROPERTY, DEFAULT_PREVIOUS_CONTEXT));
    this.nextContext = Integer.parseInt(pipelineProperties.getProperty(NEXT_CONTEXT_PROPERTY, DEFAULT_NEXT_CONTEXT));
    Log.fine("IndexExtractor previous context " + previousContext + ", next context " + nextContext);
    Log.fine("IndexExtractor max sentence length " + maxSentenceLength);
  }

  public String extractText(Document doc) {
    return extractText(doc, getIsWikipedia());
  }

  public static String extractText(Document doc, boolean isWikipedia) {
    String text = doc.get(textField(isWikipedia));
    if (text == null)
      return "";
    if (isWikipedia) {
      text = WikipediaReader.removeMarkup(StringEscapeUtils.unescapeXml(text), false, ListRemoval.COMMA);
    }
    return text.trim();
  }

  public String textField() {
    return textField(isWikipedia);
  }

  public static String textField(boolean isWikipedia) {
    // This applies to both the KBP and the Wikipedia indices
    if (!isWikipedia) {
      return KBPField.TEXT.fieldName();
    } else {
      return KBPField.WIKICONTENT.fieldName(); // XXX: this is KBBField.TEXT in
      // the older Wikipedia index!
    }
  }

  static public String docIdField() {
    return KBPField.DOCID.fieldName();
  }

  static public String datetimeField() {
    return KBPField.DATETIME.fieldName();
  }

  // validocIds should be null if you don't want to restrict the documents from
  // which the information should be extracted
  public List<CoreMap> findRelevantSentences(String entityName, EntityType entityType, int n, Set<String> validDocIds) {
    return findRelevantSentences(entityName, entityType, null, n, validDocIds);
  }

  // Lower this number to get fewer docs, but faster
  private static final int DOC_MULTIPLIER = 10;

  // validocIds should be null if you don't want to restrict the documents from
  // which the information should be extracted
  public List<CoreMap> findRelevantSentences(String entityName, EntityType entityType, Set<String> slotKeywords, int n, Set<String> validDocIds) {
    Log.fine("QUERY entity: " + entityName + " and slots: " + slotKeywords);

    Log.finest("Looking for " + DOC_MULTIPLIER * n + " docs");

    List<Integer> docIds = findRelevantDocuments(entityName, slotKeywords, DOC_MULTIPLIER * n);
    Log.fine("Found " + docIds.size() + " relevant documents for entity " + entityType + ":" + entityName + " and slots: " + slotKeywords);
    List<CoreMap> relevantSentences = findRelevantSentences(docIds, entityName, entityType, slotKeywords, n, validDocIds);
    return relevantSentences;
  }

  // validocIds should be null if you don't want to restrict the documents from
  // which the information should be extracted
  abstract List<CoreMap> findRelevantSentences(List<Integer> docIds, String entityName, EntityType entityType, Set<String> slotKeywords, int n, Set<String> validDocIds);

  List<Integer> findRelevantDocuments(String entityName, Set<String> slotKeywords, int n) {
    // Note: do this here instead of findRelevantSentences (do not
    // want to mess with the original tokens)
    entityName = QueryUtils.rewriteQueryTerm(entityName);
    if (slotKeywords != null) {
      Set<String> newKeywords = new HashSet<String>();
      for (String keyword : slotKeywords) {
        newKeywords.add(QueryUtils.rewriteQueryTerm(keyword));
      }
      slotKeywords = newKeywords;
    }

    Log.severe("Actual entity name used for doc retrieval: " + entityName);
    Query query = QueryUtils.buildSentenceQuery(entityName, textField(), slotKeywords, stopWords);
    return findRelevantDocuments(query, n);
  }

  List<Integer> findRelevantDocuments(Query query, int n) {
    // We will need at most n docs that have a hit to get n sentences
    Timing queryTimer = new Timing();
    queryTimer.start();

    Log.info("Running query " + query);

    TopDocs results;
    try {
      results = searcher.search(query, n);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    List<Integer> docIds = new ArrayList<Integer>();
    for (ScoreDoc scoreDoc : results.scoreDocs) {
      docIds.add(scoreDoc.doc);
    }

    long time = queryTimer.stop();
    totalQueryTime += time;
    totalLuceneTime += time;

    return docIds;
  }

  Document fetchDocument(int docId) {
    Timing fetchTimer = new Timing();
    fetchTimer.start();

    Document doc;
    try {
      doc = searcher.doc(docId);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    long time = fetchTimer.stop();
    totalFetchTime += time;
    totalLuceneTime += time;

    return doc;
  }

  static public List<CoreMap> sortSentences(List<CoreMap> results, ResultSortMode sortMode, Collection<String> triggerWords, int sentenceTooShort, int nerTooFew) {
    switch (sortMode) {
    case NONE:
      break;
    case NER:
      // sort by NER count...
      Collections.sort(results, new NERSorter());
      break;
    case BUCKETS:
      // this just puts the good ones at the front, but is otherwise stable
      results = reorderBuckets(results, sentenceTooShort, nerTooFew);
      break;
    case TRIGGER:
      if (triggerWords == null || triggerWords.size() == 0)
        throw new IllegalArgumentException("Asked to sort on " + sortMode + " but no trigger words were provided");
      TriggerSorter sorter = new TriggerSorter(triggerWords);
      Collections.sort(results, sorter);
      break;
    default:
      throw new AssertionError("Unknown ResultSortMode: " + sortMode);
    }
    return results;
  }

  static public class NERSorter implements Comparator<CoreMap> {
    public int compare(CoreMap first, CoreMap second) {
      int firstCount = countNER(first);
      int secondCount = countNER(second);
      return secondCount - firstCount;
    }
  }

  static public class TriggerSorter implements Comparator<CoreMap> {
    final List<Pattern> patterns;

    public TriggerSorter(Collection<String> triggerWords) {
      patterns = new ArrayList<Pattern>();
      for (String trigger : triggerWords) {
        patterns.add(Pattern.compile(StringFinder.cleanMatchRegex(trigger)));
      }
    }

    public int compare(CoreMap first, CoreMap second) {
      int firstCount = countMatches(first);
      int secondCount = countMatches(second);
      return secondCount - firstCount;
    }

    public int countMatches(CoreMap sentence) {
      return countMatches(StringFinder.toMatchString(sentence));
    }

    public int countMatches(String sentence) {
      int count = 0;
      for (Pattern pattern : patterns) {
        Matcher matcher = pattern.matcher(sentence);
        while (matcher.find())
          ++count;
      }
      return count;
    }
  }

  /**
   * Rearranges the list into two groups: those long enough with enough NER, or
   * those that don't meet one of those criteria. Is a "stable sort" within that
   * criteria.
   */
  static public List<CoreMap> reorderBuckets(List<CoreMap> results, int sentenceTooShort, int nerTooFew) {
    List<CoreMap> newResults = new ArrayList<CoreMap>();
    List<CoreMap> goodBucket = new ArrayList<CoreMap>();
    List<CoreMap> crappyBucket = new ArrayList<CoreMap>();
    for (CoreMap result : results) {
      if (sentenceLength(result) <= sentenceTooShort || countNER(result) <= nerTooFew)
        crappyBucket.add(result);
      else
        goodBucket.add(result);
    }
    newResults.addAll(goodBucket);
    newResults.addAll(crappyBucket);
    return newResults;
  }

  static public int sentenceLength(CoreMap sentence) {
    if (sentence == null)
      return 0;
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    if (tokens == null)
      return 0;
    return tokens.size();
  }

  static public int countNER(CoreMap sentence) {
    if (sentence == null)
      return 0;
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    if (tokens == null)
      return 0;
    String previousNER = Constants.NER_BLANK_STRING;
    int count = 0;
    for (int i = 0; i < tokens.size(); ++i) {
      CoreLabel word = tokens.get(i);
      String ner = word.get(NamedEntityTagAnnotation.class);
      if (ner == null || ner.equals(Constants.NER_BLANK_STRING)) {
        previousNER = Constants.NER_BLANK_STRING;
        continue;
      }
      if (ner.equals(previousNER))
        continue;
      previousNER = ner;
      ++count;
    }
    return count;
  }

  /**
   * Given a CoreMap representing a sentence, this adds to that CoreMap
   * annotations describing information about the Document the sentence was
   * extracted from. For example, it adds the path to the index, the date &amp;
   * time of the document, and the document id, assuming each of those fields
   * exist.
   */
  public CoreMap makeRelevantSentence(CoreMap sentence, Document doc) {
    if (source != null) {
      sentence.set(SourceIndexAnnotation.class, getSource());
    }
    // TODO: make up docids for stuff like the wikipedia index?
    String docId = doc.get(docIdField());
    if (docId != null) {
      sentence.set(DocIDAnnotation.class, docId);
    }
    String datetime = doc.get(datetimeField());
    if (datetime != null) {
      sentence.set(DatetimeAnnotation.class, datetime);
    }
    return sentence;
  }

  /**
   * Given a list of sentences, this combines potentially several sentences into
   * one big sentence. The TokensAnnotation and the TreeAnnotation will be
   * merged.
   */
  public CoreMap getSentenceContext(List<CoreMap> sentences, int sentenceId, Document doc) {
    if (nextContext == 0 && previousContext == 0)
      return sentences.get(sentenceId);
    int beginIndex = Math.max(sentenceId - previousContext, 0);
    int endIndex = Math.min(sentenceId + nextContext + 1, sentences.size());
    CoreMap combined = CoreMapCombiner.combine(sentences, beginIndex, endIndex, sentenceId);
    if (combined == null) {
      Log.severe("Could not get context of sentence " + sentenceId + " of document " + doc.get(docIdField()) + " in source " + getSource());
    }
    return combined;
  }

  public String luceneTime() {
    StringBuilder time = new StringBuilder();
    time.append("Total time spent in Lucene: " + totalLuceneTime / 1000 + "." + totalLuceneTime % 1000 + "s\n");
    time.append("  -- Lucene query time: " + totalQueryTime / 1000 + "." + totalQueryTime % 1000 + "s\n");
    time.append("  -- Lucene fetch time: " + totalFetchTime / 1000 + "." + totalFetchTime % 1000 + "s");
    return time.toString();
  }

  public static void main(String[] args) throws Exception {
    Log.setLevel(Level.FINE);
    Properties properties = StringUtils.argsToProperties(args);
    SentenceExtractor extractor = createIndexExtractors(ResultSortMode.NONE, properties);
    int numSentences = Integer.valueOf(properties.getProperty(DEBUG_SENTENCES_PROPERTY, "1"));

    String name = properties.getProperty(DEBUG_QUERY_PROPERTY, "Mike Quigley");

    String slotKeywordString = properties.getProperty(DEBUG_SLOT_PROPERTY, "").trim();
    List<CoreMap> sentences;
    if (slotKeywordString.equals("")) {
      sentences = extractor.findRelevantSentences(name, EntityType.ORGANIZATION, numSentences, null);
    } else {
      Set<String> slotKeywords = new HashSet<String>();
      String[] slotPieces = slotKeywordString.split(",");
      for (String piece : slotPieces) {
        slotKeywords.add(piece);
      }
      sentences = extractor.findRelevantSentences(name, EntityType.ORGANIZATION, slotKeywords, numSentences, null);
    }

    System.err.println("Found " + sentences.size() + " sentences.");
    int count = 0;
    for (CoreMap sentence : sentences) {
      System.err.print("Sentence #" + count + ": ");
      Utils.printSentence(System.err, sentence);
      count++;
    }

    System.err.println(extractor.timingInformation());
  }

  private static final Set<String> dateRelations = new HashSet<String>(Arrays.asList("org:founded", "org:dissolved", "per:date_of_birth", "per:date_of_death"));

  public static Set<String> slotKeywords(Collection<KBPSlot> slots, boolean alternateDateHandling) {
    Set<String> keywords = new HashSet<String>();
    for (KBPSlot slot : slots) {
      Set<String> queries = slotToQueries(slot, alternateDateHandling);
      for (String query : queries) {
        String trimmed = query.replaceAll("\\s+", " ").trim();
        if (trimmed.length() > 0) {
          keywords.add(trimmed);
        }
      }
    }
    return keywords;
  }

  public static Set<String> slotToQueries(KBPSlot slot, boolean alternateDateHandling) {
    Set<String> queries = new HashSet<String>();
    String query = slot.slotValue.toLowerCase();
    query = query.replaceAll(",", "");
    String preDateStripped = new String(query);
        
    // remove month names and single or double digits from dates (too specific)
    if (dateRelations.contains(slot.slotName)) {
      query = query.replaceAll("january|february|march|april|may|june|july|august|september|october|november|december", " "); // remove months
      query = query.replaceAll("\\W\\d\\d?\\W", " "); // remove days
      
      if (alternateDateHandling) {
        // include month and day as well
        String yearStripped = preDateStripped.replaceAll("[12]\\d\\d\\d", " "); // remove years
        if (yearStripped.matches(".*\\d\\d?.*")) { // make sure it still has a day
          queries.add(yearStripped);
        }
      }
    }
    
    queries.add(query);
    
    return queries;
  }

}
