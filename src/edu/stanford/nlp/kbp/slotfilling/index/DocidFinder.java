package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.lucene.analysis.Analyzer;
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

import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.StringFinder;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

public class DocidFinder {
  static public final int DOCID_QUERY_SIZE = 100;

  static public final String OFFICIAL_INDEX_PROPERTY = "index.official";

  final IndexSearcher searcher;
  final Set<String> stopWords;

  final QueryParser queryParser;

  String source = null;
  public String getSource() { return source; }
  
  public DocidFinder(String indexDir) 
    throws IOException
  {
    this(new SimpleFSDirectory(new File(indexDir)));
    source = indexDir;
  }

  public DocidFinder(Directory directory) 
    throws IOException
  {
    searcher = new IndexSearcher(directory);
    stopWords = QueryUtils.standardStopWords();

    Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_30);
    queryParser = new QueryParser(Version.LUCENE_30, 
                                  KBPField.TEXT.fieldName(), analyzer);
  }

  public String findBestDocidNoSorting(String name, String slotValue) {
    // (this is the method used in 2010)
    // TODO: This will get us some hits for 
    //  "U.S. Department of Justice"
    // but will miss documents that actually said "U.S.".  It is
    // about a 2-1 ratio of more "US" vs "U.S.".  Perhaps we should
    // search for "U.S." as well to get the rest of the hits.
    name = name.replaceAll("\\.", "");
    slotValue = slotValue.replaceAll("\\.", ""); 

    Query query = QueryUtils.buildDocidQuery(name, slotValue, 
                                             KBPField.TEXT.fieldName(), 
                                             stopWords);
    Log.info("Using query: " + query);
    return getFirstResult(query);
  }

  public String getFirstResult(Query query) {
    TopDocs results;
    try {
      results = searcher.search(query, DOCID_QUERY_SIZE);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    //System.out.println("Results found: " + results.scoreDocs.length);
    if (results != null && results.scoreDocs != null) {
      Log.severe("The above query fetched " + results.scoreDocs.length + 
                 " results.");
      for (ScoreDoc scoreDoc : results.scoreDocs) {
        int docId = scoreDoc.doc;
        Document doc;
        try {
          doc = searcher.doc(docId);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
        String docIdString = doc.get(KBPField.DOCID.fieldName());
        if (docIdString != null)
          return docIdString;
      }
    }
    return null;
  }

  public String findBestDocid(String name, String slotValue) {
    String docid = findBestDocidExactStrings(name, slotValue);
    if (docid != null)
      return docid;

    // this seems to hurt precision too much for the benefit it gives
    // recall
    //return findBestDocidInexactStrings(name, slotValue);

    return null;
  }

  public String findBestDocidInexactStrings(String name, String slotValue) {
    name = QueryUtils.rewriteQueryTerm(name);
    slotValue = QueryUtils.rewriteQueryTerm(slotValue);

    Query query = 
      QueryUtils.buildInexactDocidQuery(QueryUtils.rewriteQueryTerm(name), 
                                        QueryUtils.rewriteQueryTerm(slotValue),
                                        KBPField.TEXT.fieldName(),
                                        stopWords, true);
    Log.severe("Looking for inexact matching doc using query: " + query);
    String result = getFirstResult(query);
    if (result != null)
      return result;

    // This block gets *way* too many results, since it allows for
    // sloppy matching of any part of the query.
    //try {
    //  query = queryParser.parse(name + " " + slotValue);
    //  Log.severe("Still looking for doc, now using query: " + query);
    //  return getFirstResult(query);
    //} catch (Exception e) {
    //  // well, that didn't work for some reason
    //}
    
    return null;
  }

  /**
   * Does phrase queries for both name and slotValue.  One this is
   * done, looks for a sentence that has both terms in the same
   * sentence.  Returns that sentence if found.  Otherwise, returns
   * the first result Lucene had found.
   */
  public String findBestDocidExactStrings(String name, String slotValue) {
    // TODO: This will get us some hits for 
    //  "U.S. Department of Justice"
    // but will miss documents that actually said "U.S.".  It is
    // about a 2-1 ratio of more "US" vs "U.S.".  Perhaps we should
    // search for "U.S." as well to get the rest of the hits.
    name = QueryUtils.rewriteQueryTerm(name);
    slotValue = QueryUtils.rewriteQueryTerm(slotValue);

    Query query = QueryUtils.buildDocidQuery(name, slotValue, 
                                             KBPField.TEXT.fieldName(), 
                                             stopWords);
    Log.severe("Looking for best matching doc using query: " + query);
    TopDocs results;
    try {
      results = searcher.search(query, DOCID_QUERY_SIZE);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    //System.out.println("Results found: " + results.scoreDocs.length);
    if(results != null && results.scoreDocs != null)
      Log.severe("The above query fetched " + results.scoreDocs.length + 
                 " results.");
    
    List<Document> sameSentences = new ArrayList<Document>();
    List<Document> differentSentences = new ArrayList<Document>();

    for (ScoreDoc scoreDoc : results.scoreDocs) {
      int docId = scoreDoc.doc;
      Document doc;
      try {
        doc = searcher.doc(docId);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }      
      String docIdString = doc.get(KBPField.DOCID.fieldName());
      if (docIdString == null)
        continue;
      if (matchInSameSentence(doc, name, slotValue)) {
        sameSentences.add(doc);
      } else {
        differentSentences.add(doc);
      }
    }

    Log.severe("FOUND IN THE SAME SENTENCE: " + sameSentences.size() + 
               " of " + (sameSentences.size() + differentSentences.size()));

    if (sameSentences.size() > 0) {
      sortSameSentenceMatches(sameSentences,name, slotValue);
      return sameSentences.get(0).get(KBPField.DOCID.fieldName());
    }
    if (differentSentences.size() > 0) {
      // sadly, sorting like this doesn't seem to improve things
      //sortDifferentSentenceMatches(differentSentences, name, slotValue);
      return differentSentences.get(0).get(KBPField.DOCID.fieldName());
    }

    return null;
  }

  public boolean matchInSameSentence(Document doc, 
                                     String name, String slotValue) {
    List<String> nameList = Collections.singletonList(name);
    StringFinder nameFinder = new StringFinder(nameList);
    List<String> slotList = Collections.singletonList(slotValue);
    StringFinder slotFinder = new StringFinder(slotList);

    Annotation annotatedText = 
      LucenePipelineCacher.getAnnotationFromDocNoExceptions(doc);
    if (annotatedText == null) {
      return false;
    }
    
    List<CoreMap> sentences = annotatedText.get(SentencesAnnotation.class);
    for (CoreMap sentence : sentences) {
      if (slotFinder.matches(sentence) && nameFinder.matches(sentence)) {
        List<Pair<Integer, Integer>> nameMatches = 
          nameFinder.whereItMatches(sentence);
        List<Pair<Integer, Integer>> slotMatches =
          slotFinder.whereItMatches(sentence);
        if (name.length() >= slotValue.length()) {
          slotMatches = filterOverlapping(slotMatches, nameMatches);
        } else {
          nameMatches = filterOverlapping(nameMatches, slotMatches);
        }
        if (slotMatches.size() > 0 && nameMatches.size() > 0) {
          Log.severe("Found non-overlapping matches for " +
                     StringFinder.toMatchString(sentence));
          return true;
        } else {
          Log.severe("Rejecting overlapping matches for " + 
                     StringFinder.toMatchString(sentence));          
        }
      }
    }

    return false;
  }

  public List<Pair<Integer, Integer>>
    filterOverlapping(List<Pair<Integer, Integer>> originalMatches,
                      List<Pair<Integer, Integer>> filterMatches) 
  {
    List<Pair<Integer, Integer>> filtered = 
      new ArrayList<Pair<Integer, Integer>>();
    int originalIndex = 0;
    int filterIndex = 0;
    while (originalIndex < originalMatches.size() &&
           filterIndex < filterMatches.size()) {
      int originalBegin = originalMatches.get(originalIndex).first();
      int originalEnd = originalMatches.get(originalIndex).second();
      int filterBegin = filterMatches.get(filterIndex).first();
      int filterEnd = filterMatches.get(filterIndex).second();
      if (filterEnd < originalEnd) {
        ++filterIndex;
        continue;
      }
      // now we know filterEnd >= originalEnd
      if (filterBegin <= originalBegin) {
        // here we have that 
        // filterBegin <= originalBegin && filterEnd >= originalEnd
        // so the filter has eclipsed the original...
        ++originalIndex;
        continue;
      }
      // filterBegin > originalBegin, so no filter eclipsed the
      // original and this is an unblocked original segment
      filtered.add(originalMatches.get(originalIndex));
      ++originalIndex;
    }    
    for ( ; originalIndex < originalMatches.size(); ++originalIndex) {
      filtered.add(originalMatches.get(originalIndex));
    }
    return filtered;
  }

  public void sortWithKey(List<Document> docs,
                          final Map<Document, Double> scores) {
    Collections.sort(docs, new Comparator<Document>() {
        public int compare(Document d1, Document d2) {
          double s1 = scores.get(d1);
          double s2 = scores.get(d2);
          if (s1 < s2)
            return -1;
          if (s1 > s2)
            return 1;
          return 0;
        }
      });    
  }

  public void sortSameSentenceMatches(List<Document> docs,
                                      String name, String slotValue) {
    Map<Document, Double> scores = new HashMap<Document, Double>();
    for (Document doc : docs) {
      scores.put(doc, matchSameSentenceScore(doc, name, slotValue));
    }
    sortWithKey(docs, scores);
  }

  public void sortDifferentSentenceMatches(List<Document> docs,
                                           String name, String slotValue) {
    final Map<Document, Double> scores = new HashMap<Document, Double>();
    for (Document doc : docs) {
      scores.put(doc, matchDifferentSentenceScore(doc, name, slotValue));
    }
    sortWithKey(docs, scores);
  }

  public double matchSameSentenceScore(Document doc,
                                       String name, String slotValue) {
    List<String> nameList = Collections.singletonList(name);
    StringFinder nameFinder = new StringFinder(nameList);
    List<String> slotList = Collections.singletonList(slotValue);
    StringFinder slotFinder = new StringFinder(slotList);

    Annotation annotatedText = 
      LucenePipelineCacher.getAnnotationFromDocNoExceptions(doc);
    if (annotatedText == null) {
      return Double.MAX_VALUE;
    }
    
    List<CoreMap> sentences = annotatedText.get(SentencesAnnotation.class);
    double bestScore = Double.MAX_VALUE;
    for (CoreMap sentence : sentences) {
      if (slotFinder.matches(sentence) && nameFinder.matches(sentence)) {
        List<Pair<Integer, Integer>> nameMatches = 
          nameFinder.whereItMatches(sentence);
        List<Pair<Integer, Integer>> slotMatches =
          slotFinder.whereItMatches(sentence);
        if (name.length() >= slotValue.length()) {
          slotMatches = filterOverlapping(slotMatches, nameMatches);
        } else {
          nameMatches = filterOverlapping(nameMatches, slotMatches);
        }
        if (slotMatches.size() == 0 || nameMatches.size() == 0) {
          continue;
        }
        for (int i = 0; i < nameMatches.size(); ++i) {
          for (int j = 0; j < slotMatches.size(); ++j) {
            int nameBegin = nameMatches.get(i).first();
            int nameEnd = nameMatches.get(i).second();
            int slotBegin = slotMatches.get(j).first();
            int slotEnd = slotMatches.get(j).second();
            int score = Math.min(Math.abs(nameBegin - slotEnd),
                                 Math.abs(nameEnd - slotBegin));
            Log.severe("Same sentence match score for " + 
                       StringFinder.toMatchString(sentence) + score);
            if (score < bestScore) {
              bestScore = score;
            }
          }
        }
      }
    }
    if (bestScore < 0)
      return Double.MAX_VALUE;
    return bestScore;
  }

  public double matchDifferentSentenceScore(Document doc,
                                            String name, String slotValue) {
    List<String> nameList = Collections.singletonList(name);
    StringFinder nameFinder = new StringFinder(nameList);
    List<String> slotList = Collections.singletonList(slotValue);
    StringFinder slotFinder = new StringFinder(slotList);

    Annotation annotatedText = 
      LucenePipelineCacher.getAnnotationFromDocNoExceptions(doc);
    if (annotatedText == null) {
      return 0.0;
    }
    
    List<CoreMap> sentences = annotatedText.get(SentencesAnnotation.class);

    int countName = 0;
    int countSlot = 0;
    int lastName = -1;
    int lastSlot = -1;
    int closestPair = -1;
    int sentenceNum = 0;
    for (CoreMap sentence : sentences) {
      if (nameFinder.matches(sentence)) {
        lastName = sentenceNum;
        ++countName;
      }
      if (slotFinder.matches(sentence)) {
        lastSlot = sentenceNum;
        ++countSlot;
      }

      if (lastName >= 0 && lastSlot >= 0) {
        if (closestPair < 0 || Math.abs(lastName - lastSlot) < closestPair) {
          closestPair = Math.abs(lastName - lastSlot);
        }
      }

      ++sentenceNum;
    }
    if (countName == 0 || countSlot == 0 || closestPair < 0) {
      return 0.0;
    }
    return closestPair - Math.log(countName + countSlot);
  }
  
  public int getDocumentFrequency(String slotValue) throws IOException {
    slotValue = slotValue.replaceAll("\\.", ""); 
    
    Query q = QueryUtils.buildPhraseQuery(slotValue, KBPField.TEXT.fieldName(), new HashSet<String>());
    TopDocs topDocs = searcher.search(q, 1);
    return topDocs.totalHits;
  }

  static public void main(String[] args) 
    throws Exception
  {
    DocidFinder finder = new DocidFinder(args[0]);
    String docid = finder.findBestDocid(args[1], args[2]);
    System.out.println(docid);
  }
}