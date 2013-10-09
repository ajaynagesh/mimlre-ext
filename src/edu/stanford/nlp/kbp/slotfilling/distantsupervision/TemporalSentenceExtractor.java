package edu.stanford.nlp.kbp.slotfilling.distantsupervision;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.queryParser.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.SimpleFSDirectory;
import org.apache.lucene.util.Version;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.Utils;
import edu.stanford.nlp.kbp.slotfilling.index.LucenePipelineCacher;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.NumberAnnotator;
import edu.stanford.nlp.pipeline.QuantifiableEntityNormalizingAnnotator;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.time.SUTime;
import edu.stanford.nlp.time.TimeAnnotations.TimexAnnotations;
import edu.stanford.nlp.time.TimeAnnotator;
import edu.stanford.nlp.time.TimeExpression;
import edu.stanford.nlp.util.CoreMap;

public class TemporalSentenceExtractor extends EntitySentenceExtractor {
  StanfordCoreNLP pipeline;
  String indexPath = "/scr/nlp/data/tackbp2010/indices/lr_en_100622_2000_Index_Cached";
  QueryParser parser;
  IndexSearcher searcher;

  @SuppressWarnings("deprecation")
  public TemporalSentenceExtractor(int maxIndexSentences) throws Exception {
    super(maxIndexSentences, 0);
    Properties props = new Properties();
    //props.setProperty("annotators", "tokenize, ssplit, pos, lemma, parse, ner, regexner, dcoref");
    props.setProperty("annotators", "regexner, dcoref");
    pipeline = new StanfordCoreNLP(props, false);
    pipeline.addAnnotator(new NumberAnnotator(false));
    pipeline.addAnnotator(new QuantifiableEntityNormalizingAnnotator(false, false));
    pipeline.addAnnotator(new TimeAnnotator());

    Directory directory = new SimpleFSDirectory(new File(indexPath));
    searcher = new IndexSearcher(directory);
    StandardAnalyzer analyzer = new StandardAnalyzer(Version.LUCENE_CURRENT);
    parser = new QueryParser(Version.LUCENE_CURRENT, "title", analyzer);
    parser.setDefaultOperator(QueryParser.Operator.AND);
  }

  /**
   * Find articles that have a wiki URL field in Freebase
   */
  @SuppressWarnings("unchecked")
  public List<CoreMap> findSentences(KBPEntity entity, Set<String> knownSlots, File sourceFile, boolean testMode, Set<String> validDocIds)
      throws Exception {
    
    if(validDocIds != null){
      throw new Exception("using valid doc ids not implemented in this class");
    }
    List<CoreMap> foundSentences = new ArrayList<CoreMap>();
    try {
      // PrintStream posStart = new PrintStream("posStart.txt");
      // PrintStream posEnd = new PrintStream("posEnd.txt");

      String title = entity.name;
      if (title != null) {
        title = QueryParser.escape(title);
        Query q = parser.parse(title);
        TopDocs docs = searcher.search(q, 10);
        if (docs.scoreDocs == null || docs.scoreDocs.length == 0) {
          return foundSentences;
        }
        ScoreDoc scoredoc = docs.scoreDocs[0];
        int docId = scoredoc.doc;
        Document doc = searcher.doc(docId);
        if (doc == null) {
          logger.fine("Doc " + docId + " not found");
        } else {
          CoreMap annotatedText = LucenePipelineCacher.getAnnotationFromDoc(doc);
          List<CoreMap> sentences = annotatedText.get(SentencesAnnotation.class);

          for (CoreMap sentence : sentences) {

            List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
            String sentenceStr = Utils.sentenceToMinimalString(sentence);

            sentence.set(CoreAnnotations.TextAnnotation.class, sentenceStr);
            sentence.set(CoreAnnotations.SentencesAnnotation.class, Arrays.asList(sentence));
            Annotation aa = (Annotation) sentence;
            // Annotation aa = new Annotation(sentenceStr);
            pipeline.annotate(aa);
            for (Class s : sentence.keySet()) {
              System.out.println("old has class " + s);
            }
            // List<CoreLabel> newTokens = new ArrayList<CoreLabel>();
            int j = 0;
            for (CoreLabel t : sentence.get(CoreAnnotations.TokensAnnotation.class)) {

              t.set(CoreAnnotations.TokenBeginAnnotation.class, j);
              t.set(CoreAnnotations.TokenEndAnnotation.class, j + 1);
              j++;

            }

            for (Class s : aa.keySet()) {
              System.out.println("has class " + s);
            }
            // for (CoreLabel t :
            // aa.get(CoreAnnotations.TokensAnnotation.class)) {
            // // for (Class s : t.keySet()) {
            // // System.out.println("new token has class " + s);
            // //
            // // }
            // System.out.println("for sentence " + sentenceStr +
            // "tokenbegin is "
            // + t.get(CoreAnnotations.TokenBeginAnnotation.class) +
            // " and end annotation is "
            // + t.get(CoreAnnotations.TokenEndAnnotation.class));
            // // break;
            // }
            List<CoreMap> cms = aa.get(TimexAnnotations.class);
            for (CoreMap cm : cms) {

              TimeExpression tm = cm.get(TimeExpression.Annotation.class);
              SUTime.Temporal t = tm.getTemporal();

              List<String> dateNormalizedTokens = new ArrayList<String>();

              int beginToken = cm.get(CoreAnnotations.TokenBeginAnnotation.class);
              int endToken = cm.get(CoreAnnotations.TokenEndAnnotation.class);
              String dateVal = t.getTimexValue();

              for (int i = 0; i < tokens.size(); i++) {
                if (i == beginToken)
                  dateNormalizedTokens.add(dateVal);
                else if (i > beginToken && i <= endToken)
                  continue;
                else
                  dateNormalizedTokens.add(tokens.get(i).word());
              }

              for (String dateSlot : knownSlots) {
                if (matchSlotInSentence(dateSlot, dateNormalizedTokens, new String[] { dateSlot }, new ArrayList<Span>())) {
                  foundSentences.add(aa.get(CoreAnnotations.SentencesAnnotation.class).get(0));
                }
              }
            }
          }
          // numMatchingDocs++;
          // for (Fieldable field : doc.getFields()) {
          // if (field.name().equals("title"))
          // System.out.println(field.name() + doc.get(field.name()));

          // }

        }
      }

      // System.out.println("\nNum matching docs " + numMatchingDocs +
      // " out of " + numAllDocs);
      // IOUtils.writeObjectToFile(allTempObjs, "allTempObjs.ser");
      return foundSentences;
    } catch (Exception e) {
      e.printStackTrace();
    }
    return foundSentences;
  }

  /**
   * Finds all token spans where this slot matches over the sentence tokens This
   * implements all out approximate match heuristics
   * 
   * @param slot
   * @param sentence
   */
  static public boolean matchSlotInSentence(String slot, List<String> dateNormalizedTokens,
      String[] slotValueTokens, List<Span> matchingSpans) {

    boolean matchedAny = false;
    // this stores all matches in this sentence. must be reset because this is
    // called for multiple sentences
    matchingSpans.clear();
    List<Boolean> matchingSpanExact = new ArrayList<Boolean>();

    List<String[]> names = new ArrayList<String[]>();
    names.add(slotValueTokens);
    // if (slot.alternateSlotValues != null)
    // names.addAll(slot.alternateSlotValues);

    boolean[] used = new boolean[dateNormalizedTokens.size()];
    Arrays.fill(used, false);
    boolean exact = true;
    for (String[] name : names) {
      for (int start = 0; start < dateNormalizedTokens.size();) {
        if (used[start]) { // already taken by another name variant
          start++;
          continue;
        }
        if (nameMatches(name, dateNormalizedTokens, start)) {
          matchedAny = true;
          logger.fine("MATCHED " + (exact ? "exact" : "alternate") + " slot " + slot + ":" + slot + " at position "
              + start);
          matchingSpans.add(new Span(start, start + name.length));
          matchingSpanExact.add(exact);
          for (int i = 0; i < name.length; i++) {
            used[start + i] = true;
          }
          start += name.length;
        } else {
          start++;
        }
      }
      exact = false;
    }
    return matchedAny;
  }

  static public boolean nameMatches(String[] name, List<String> dateNormalizedTokens, int start) {

    if (start + name.length > dateNormalizedTokens.size())
      return false;
    for (int i = 0; i < name.length; i++) {
      if (Constants.CASE_INSENSITIVE_SLOT_MATCH) {
        if (!name[i].equalsIgnoreCase(dateNormalizedTokens.get(start + i))) {
          return false;
        }
      } else {
        if (!name[i].equals(dateNormalizedTokens.get(start + i))) {
          return false;
        }
      }
    }
    return true;
  }

  // public List<CoreMap> findSentences(KBPEntityMention entity, Set<String>
  // knownSlots, boolean testMode)
  // throws Exception {
  //
  // List<CoreMap> sentences = new ArrayList<CoreMap>();
  // Properties props = new Properties();
  // props.setProperty("annotators",
  // "tokenize, ssplit, pos, lemma, ner, regexner, parse, dcoref");
  //
  // StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
  // pipeline.addAnnotator(new NumberAnnotator(false));
  // pipeline.addAnnotator(new QuantifiableEntityNormalizingAnnotator(false,
  // false));
  //
  // Annotation a = new Annotation("John was born on 1985-08-08.");
  //
  // pipeline.annotate(a);
  // for (CoreMap sentence : a.get(CoreAnnotations.SentencesAnnotation.class))
  // sentences.add(sentence);
  // return sentences;
  // }

  // void a() {
  // System.out.println("inside find sentences");
  // List<CoreMap> sentences = new ArrayList<CoreMap>();
  // Properties props = new Properties();
  // // props.setProperty("annotators",
  // "tokenize, ssplit, pos, lemma, ner, regexner, parse, dcoref");
  // props.setProperty("annotators",
  // "tokenize, ssplit, pos, lemma, ner, regexner, parse, dcoref");
  // StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
  // pipeline.addAnnotator(new NumberAnnotator(false));
  // pipeline.addAnnotator(new QuantifiableEntityNormalizingAnnotator(false,
  // false));
  //
  // Annotation a = new Annotation("Obama is the president.");
  // pipeline.annotate(a);
  //
  // sentences.add(a);

  // }

  public static void main(String[] args) {
    try {
      @SuppressWarnings("unused")
      TemporalSentenceExtractor r = new TemporalSentenceExtractor(500);
    } catch (Exception e) {
      e.printStackTrace();
    }
    // r.a();
  }

}
