package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.File;
import java.io.IOException;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.SimpleFSDirectory;
import org.apache.lucene.util.Version;


/**
 * This would live elsewhere if we had anywhere in particular to put
 * Lucene utilities
 *
 * @author John Bauer
 */
public class IndexCombiner {
  /**
   * Private constructor... static methods only
   */
  private IndexCombiner() {}
  
  @SuppressWarnings("deprecation")
  static void combineSequentially(Directory outputDirectory,
                                  Directory ... inputDirectories) 
    throws IOException
  {
    IndexWriter writer = 
      new IndexWriter(outputDirectory,
                      new StandardAnalyzer(Version.LUCENE_CURRENT), 
                      true, IndexWriter.MaxFieldLength.LIMITED);
    
    for (Directory inputDirectory : inputDirectories) {
      System.out.println("Processing " + inputDirectory);
      IndexSearcher searcher = new IndexSearcher(inputDirectory);
      int numDocs = searcher.maxDoc();

      for (int docId = 0; docId < numDocs; ++docId) {
        Document doc = searcher.doc(docId);
        
        writer.addDocument(doc);        
      }
      searcher.close();
      System.out.println("Done with " + inputDirectory);
    }

    System.out.println("Finishing index...");
    writer.optimize();
    writer.close();
    System.out.println("Done!");
  }

  static void combineSequentially(String outputPath,
                                  String ... inputPaths)
    throws IOException
  {
    Directory[] inputDirectories = new Directory[inputPaths.length];
    for (int i = 0; i < inputPaths.length; ++i) {
      inputDirectories[i] = new SimpleFSDirectory(new File(inputPaths[i]));
    }
    combineSequentially(new SimpleFSDirectory(new File(outputPath)),
                        inputDirectories);
  }

  /**
   * Combines a set of Lucene indices into a single index that contains all
   * Usage: edu.stanford.nlp.kbp.index.IndexCombiner inputIndex1 inputIndex2 ... outputIndex
   * @param args
   * @throws IOException
   */
  public static void main(String[] args) 
    throws IOException
  {
    String outputDirectory = args[args.length - 1];
    String[] inputDirectory = new String[args.length - 1];
    for (int i = 0; i < args.length - 1; ++i) {
      inputDirectory[i] = args[i];
    }

    // TODO: this block appears in 3 programs now... factor it out
    File file = new File(outputDirectory);
    if (file.exists() && !file.isDirectory()) {
      System.out.println("Hey, be careful, you almost overwrote an " +
                         "important file: " + outputDirectory);
      System.exit(1);
    } else if (file.exists()) {
      System.out.println("There's already something here: " + outputDirectory);
      System.out.println("Try running with a directory that doesn't " +
                         "already exist; the directory will be created");
      System.exit(1);
    }    
    
    combineSequentially(outputDirectory, inputDirectory);
  }
}
