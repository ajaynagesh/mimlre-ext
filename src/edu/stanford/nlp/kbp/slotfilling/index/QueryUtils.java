package edu.stanford.nlp.kbp.slotfilling.index;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;

import edu.stanford.nlp.kbp.slotfilling.common.Log;

public class QueryUtils {
  private QueryUtils() {}; // static methods only

  static public Set<String> standardStopWords() {
    Set<String> stopWords = new HashSet<String>();
    for (Object word : StandardAnalyzer.STOP_WORDS_SET) {
      if (word instanceof String) {
        stopWords.add((String) word);
      } else {
        throw new RuntimeException("StandardAnalyzer.STOP_WORDS_SET member " +
                                   word + " is not a String");
      }
    }
    return stopWords;
  }

  static public Query buildPhraseQuery(String entityName, 
                                       String field,
                                       Set<String> stopWords) {
    PhraseQuery phraseQuery = new PhraseQuery();
    String[] pieces = entityName.split(" +");
    int position = 0;
    for (String piece : pieces) {
      piece = piece.toLowerCase();
      if (!stopWords.contains(piece)) {
        phraseQuery.add(new Term(field, piece.toLowerCase()), position);
      }
      ++position;
    }
    return phraseQuery;
  }

  static public Query buildSentenceQuery(String entityName, 
                                         String field,
                                         Set<String> slotKeywords,
                                         Set<String> stopWords) {
    Query phraseQuery = buildPhraseQuery(entityName, field, stopWords);
    if (slotKeywords == null || slotKeywords.size() == 0)
      return phraseQuery;

    BooleanQuery finalQuery = new BooleanQuery();
    BooleanQuery keywordQuery = new BooleanQuery();
    for (String keyword : slotKeywords) {
      keywordQuery.add(buildPhraseQuery(keyword, field, stopWords), 
                       BooleanClause.Occur.SHOULD);
    }
    finalQuery.add(phraseQuery, BooleanClause.Occur.MUST);
    finalQuery.add(keywordQuery, BooleanClause.Occur.MUST);
    return finalQuery;
  }

  static public Query buildDocidQuery(String name, String slotValue,
                                      String field, Set<String> stopWords) {
    BooleanQuery finalQuery = new BooleanQuery();
    Log.finest("Using stop words: " + stopWords);
    Log.finest("Adding to query: " + name);
    finalQuery.add(buildPhraseQuery(name, field, stopWords),
                   BooleanClause.Occur.MUST);
    Log.finest("Adding to query: " + slotValue);
    finalQuery.add(buildPhraseQuery(slotValue, field, stopWords),
                   BooleanClause.Occur.MUST);
    return finalQuery;
  }

  static public Query buildInexactDocidQuery(String name, String slotValue,
                                             String field, 
                                             Set<String> stopWords,
                                             boolean mustOccur) {
    BooleanQuery finalQuery = new BooleanQuery();
    Log.finest("Building inexact docid query using stop words: " + 
               stopWords);
    BooleanClause.Occur occur = 
      ((mustOccur) ? BooleanClause.Occur.MUST : BooleanClause.Occur.SHOULD);
    List<String> words = new ArrayList<String>();
    words.addAll(Arrays.asList(name.split(" +")));
    words.addAll(Arrays.asList(slotValue.split(" +")));
    for (String word : words) {
      word = word.toLowerCase();
      if (!stopWords.contains(word)) {
        finalQuery.add(new TermQuery(new Term(field, word)), occur);
      }
    }
    return finalQuery;
  }

  static public String rewriteQueryTerm(String query) {
    query = query.replaceAll("\\.", "");
    
    // these suffixes are affected by the dot removal => search docs
    // without them
    // TODO: is this the best way to do it?
    if(query.endsWith(" Corp")){
      query = query.substring(0, query.length() - 4).trim();
    } else if(query.endsWith(" Co")){
      query = query.substring(0, query.length() - 2).trim();
    }
    
    // lucene doesn't like dashes
    query = query.replaceAll("\\-", " ");
    // or commas
    query = query.replaceAll("\\,", " ");
    
    // common but special characters
    query = query.replaceAll("&amp;amp;", "&");
    query = query.replaceAll("&amp;", "&");
    return query;
  }

}
