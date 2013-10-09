package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.File;
import java.io.IOException;

import org.apache.lucene.document.Document;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.SimpleFSDirectory;
import org.apache.lucene.index.CorruptIndexException;

public class PrintDocumentFromId {
  IndexSearcher searcher;

  private PrintDocumentFromId(String path) 
    throws CorruptIndexException, IOException
  {
    this(new SimpleFSDirectory(new File(path)));
  }

  private PrintDocumentFromId(Directory inputDirectory) 
    throws CorruptIndexException, IOException
  {
    searcher = new IndexSearcher(inputDirectory);
  }

  private Document getDoc(int docId) 
    throws CorruptIndexException, IOException
  {
    return searcher.doc(docId);
  }

  /**
   * To run: the args should be
   * <br>
   * path id id id ...
   */
  static public void main(String[] args) 
    throws CorruptIndexException, IOException
  {
    if (args.length < 2) {
      System.out.println("The arguments should be:");
      System.out.println("  <index path> <docid>...");
      System.exit(2);
    }
    PrintDocumentFromId printer = new PrintDocumentFromId(args[0]);
    for (int i = 1; i < args.length; ++i) {
      int docId = Integer.valueOf(args[i]);
      System.out.println("================================");
      System.out.println(" Dumping document " + docId);
      System.out.println("--------------------------------");
      System.out.println(printer.getDoc(docId));
    }
  }
}
