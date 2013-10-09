package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.SimpleFSDirectory;

import edu.stanford.nlp.kbp.slotfilling.common.AntecedentGenerator;
import edu.stanford.nlp.kbp.slotfilling.common.EntityType;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.SourceIndexDocIDAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Timing;

public class CachedIndexExtractor extends IndexExtractor {
  long totalExtractTime = 0;
  long totalProcessingTime = 0;

  static final String EXTRA_RESULTS_PROPERTY = "index.extraresults.factor";
  static final String EXTRA_RESULTS_DEFAULT = "1.0";
  final double extraResultsFactor;

  public CachedIndexExtractor(String indexDir, ResultSortMode sortMode, Properties pipelineProperties) throws IOException {
    this(new SimpleFSDirectory(new File(indexDir)), sortMode, pipelineProperties);
    setSource(indexDir);
    Log.fine("Constructed a CachedIndexExtractor pointing to " + indexDir);
    Log.fine("Using sort mode " + sortMode + " for index " + indexDir);
    Log.fine("Extra results factor " + extraResultsFactor + " for index " + indexDir);
  }

  public CachedIndexExtractor(Directory directory, ResultSortMode sortMode, Properties pipelineProperties) throws IOException {
    super(directory, sortMode, pipelineProperties);
    extraResultsFactor = Double.parseDouble(pipelineProperties.getProperty(EXTRA_RESULTS_PROPERTY, EXTRA_RESULTS_DEFAULT));
  }

  List<CoreMap> findRelevantSentences(List<Integer> docIds, String entityName, EntityType entityType, Set<String> slotKeywords, int numResults, Set<String> validDocIds) {
    List<CoreMap> relevantSentences = new ArrayList<CoreMap>();
    int failedDocCount = 0;
    int docCount = 0;
    Set<String> seenDocIDs = new HashSet<String>();
    for (Integer luceneDocId : docIds) {

      if (relevantSentences.size() >= numResults || (validDocIds != null && seenDocIDs.equals(validDocIds)))
        break;

      Document doc = fetchDocument(luceneDocId);

      Log.fine("Starting doc " + docCount);

      Timing extractTimer = new Timing();
      extractTimer.start();
      Annotation annotatedText = null;
      try {
        annotatedText = LucenePipelineCacher.getAnnotationFromDoc(doc);
      } catch (Exception e) {
        // throw new RuntimeException(e);
        // we do throw exceptions on some really uncommon and weird texts. Let's
        // keep this robust and try to continue.
        failedDocCount++;
        Log.fine("WARNING: failed to read annotation from index due to exception below. Continuing...");
        e.printStackTrace();
      }
      totalExtractTime += extractTimer.stop();

      Field docId = doc.getField(docIdField());

      // Log.severe("validdocids inside cachedIndexExtractor are " +
      // StringUtils.join(validDocIds, ";"));

      if (validDocIds != null) {
        if (docId == null || !validDocIds.contains(docId.stringValue().trim())) {
          Log.fine("Document not a part of valid doc ids.");
          ++docCount;
          continue;
        } else
          seenDocIDs.add(docId.stringValue().trim());
      }
      if (annotatedText == null) {
        String docIdText = (docId != null ? (" (" + docId.stringValue() + ")") : " (unknown docid)");
        Log.fine("Doc " + docCount + docIdText + " did not have an annotation; ignoring");
        ++docCount;
        continue;
      }

      Timing processTime = new Timing();
      processTime.start();
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
        CoreMap relevantSentence = makeRelevantSentence(sentenceContext, doc);
        relevantSentence.set(SourceIndexDocIDAnnotation.class, luceneDocId);
        relevantSentences.add(relevantSentence);
        if (relevantSentences.size() >= numResults * extraResultsFactor)
          break;
      }
      totalProcessingTime += processTime.stop();

      if (failedDocCount > 0) {
        Log.fine("Retrieval failed on " + failedDocCount + " out of " + docCount + " documents for entity " + entityName);
      }
      Log.fine("================================");
      ++docCount;
    }

    List<CoreMap> sorted = sortSentences(relevantSentences, sortMode, slotKeywords, sentenceTooShort, nerTooFew);
    if (sorted.size() > numResults) {
      // new list so that we don't keep pointers to all of the other
      // CoreMaps (via a sublist of a larger list full of unused CoreMaps)
      return new ArrayList<CoreMap>(sorted.subList(0, numResults));
    } else {
      return sorted;
    }
  }

  public String timingInformation() {
    StringBuilder time = new StringBuilder();
    time.append(luceneTime());
    time.append("\n");
    time.append("Total time spent extracting the annotations: " + totalExtractTime / 1000 + "." + totalExtractTime % 1000 + "s\n");
    time.append("Total time spent processing: " + totalProcessingTime / 1000 + "." + totalProcessingTime % 1000 + "s\n");
    return time.toString();
  }

  /**
   * Scans this index and verifies that every document has a serialized
   * Annotation that can be read
   * 
   * @throws IOException
   * @throws CorruptIndexException
   * @throws ClassNotFoundException
   */
  public void verifyAnnotations() throws CorruptIndexException, IOException, ClassNotFoundException {
    IndexReader ir = searcher.getIndexReader();
    int maxDocs = ir.maxDoc();
    System.out.printf("Found %d documents in the given index.\n", maxDocs);
    int failed = 0;
    for (int i = 0; i < maxDocs; i++) {
      Document doc = ir.document(i);
      String text = extractText(doc);
      try {
        StringBuilder sb = new StringBuilder();
        LucenePipelineCacher.getAnnotationFromDoc(doc, sb);
        System.err.println("ANNOTATION:\n" + sb.toString());
      } catch (Exception e) {
        System.err.println("DOCUMENT FAILED (exception printed afterwards):\n" + text);
        System.err.println("The annotation string that failed is:\n" + LucenePipelineCacher.getAnnotationStringFromDoc(doc));
        System.err.println("Exception thrown for the above annotation:");
        e.printStackTrace();
        failed++;
      }
      assert (doc != null);
      if (i % 100 == 0)
        System.out.print(".");
    }
    System.out.println();
    System.out.printf("Processed %d documents. %d had incorrect annotations.\n", maxDocs, failed);
  }

  /**
   * The main method scans the given index and verifies that every document has
   * a serialized Annotation that can be read
   * 
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception {
    CachedIndexExtractor iex = new CachedIndexExtractor(args[0], ResultSortMode.NONE, new Properties());
    iex.verifyAnnotations();
  }
}