package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.SimpleFSDirectory;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationSerializer;
import edu.stanford.nlp.util.Pair;

public class SimpleDocumentSearch {
  static final String TEXT_FIELD = "text";
  
  public static void main(String[] args) throws Exception {
    // this must point to a valid KBP index
    String index = "/scr/nlp/data/tackbp2010/indices/lr_en_100622_2000_Index_Cached";
    // this is your query 
    String query = "George Bush"; 
    // max number of docs to return
    int maxDocs = 10;
    
    // build a Boolean query
    BooleanQuery keywordQuery = new BooleanQuery();
    for (String keyword : query.split("\\s+")) {
      PhraseQuery pq = new PhraseQuery();
      pq.add(new Term(TEXT_FIELD, keyword.toLowerCase()));
      keywordQuery.add(pq, BooleanClause.Occur.SHOULD);
    }

    // extract the top docs and their annotations from the index
    Directory indexDir = new SimpleFSDirectory(new File(index));
    IndexSearcher searcher = new IndexSearcher(indexDir);
    TopDocs results = searcher.search(keywordQuery, maxDocs);
    List<Pair<Document, Annotation>> docs = new ArrayList<Pair<Document,Annotation>>();
    for (ScoreDoc scoreDoc : results.scoreDocs) {
      Document doc = searcher.doc(scoreDoc.doc);
      String annotationBytes = doc.get("coremap");
      ByteArrayInputStream byteInput = 
           new ByteArrayInputStream(annotationBytes.getBytes("ISO-8859-1"));
      AnnotationSerializer ser = 
           new KBPAnnotationSerializer(true, true);
      Annotation annotation = ser.load(byteInput);
      System.err.println("\tFound docid " + scoreDoc.doc + " with score " + scoreDoc.score + 
          ". It contains " + annotation.get(SentencesAnnotation.class).size() + " sentences.");
      docs.add(new Pair<Document, Annotation>(doc, annotation));
    }
    
    
  }    
}
