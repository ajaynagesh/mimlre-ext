package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.File;

import org.apache.lucene.document.Document;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.SimpleFSDirectory;

/**
 * Quick sanity check for a cached index: does each document contain both text and annotation?
 */
public class VerifyCachedIndex {
  public static void main(String[] args) throws Exception {
    IndexSearcher searcher = new IndexSearcher(new SimpleFSDirectory(new File(args[0])));
    boolean isWikipedia = false;
    int numDocs = searcher.maxDoc();
    int countNoText = 0, countNoAnno = 0, count = 0;
    for (int i = 0; i < numDocs; ++i) {
      Document doc = searcher.doc(i);
      count ++;
      String docid = doc.get(KBPField.DOCID.fieldName());
      String text = doc.get(IndexExtractor.textField(isWikipedia));
      String annotation = doc.get(KBPField.COREMAP.fieldName());
      if(annotation == null || annotation.length() == 0){
        countNoAnno ++;
      }
      if(text != null && text.length() != 0){
        if(annotation == null || annotation.length() == 0){
          System.err.println("Document " + docid + " has no annotation!");
        }
      } else {
        countNoText ++;
      }
      if(count % 1000 == 0){
        System.err.println(count + " " + countNoText + " " + countNoAnno);
      }
    }
  }
}
