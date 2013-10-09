package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.List;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.SimpleFSDirectory;

import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationSerializer;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;

/**
 * Output corpus wide annotation statistics for a CoreNLP-processed Lucene index.
 * 
 * Output is a TSV file.
 * 
 * @author Spence Green
 *
 */
public final class IndexAnnotationGrossStatistics {

  private static enum AnnotationType {Word, Tag};
  
  private static final Counter<String> counter = new ClassicCounter<String>();

  // this must point to a valid KBP index
  private static final String indexDirPath = "/scr/nlp/data/tackbp2010/indices/lr_en_100622_2000_Index_Cached";
  
  private IndexAnnotationGrossStatistics() {}
  
  private static void statsFromIndex(AnnotationType annotationType) {
    try {
      Directory indexDir = new SimpleFSDirectory(new File(indexDirPath));
      IndexSearcher searcher = new IndexSearcher(indexDir);
      int maxDocId = searcher.maxDoc();
      for (int i = 0; i < maxDocId; ++i) {
        Document doc = searcher.doc(i);
        
        String annotationBytes = doc.get("coremap");
        ByteArrayInputStream byteInput = 
            new ByteArrayInputStream(annotationBytes.getBytes("ISO-8859-1"));
        AnnotationSerializer ser = 
            new KBPAnnotationSerializer(true, true);
        Annotation annotation = ser.load(byteInput);

        List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
        for (CoreMap sentence : sentences) {
          for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
            if (annotationType == AnnotationType.Word) {
              String word = token.get(TextAnnotation.class);
              counter.incrementCount(word);
            } else if (annotationType == AnnotationType.Tag) {
              String pos = token.get(PartOfSpeechAnnotation.class);
              counter.incrementCount(pos);
            } else {
              throw new RuntimeException();
            }
          }
        }
      }
    } catch (CorruptIndexException e) {
      e.printStackTrace();
    } catch (UnsupportedEncodingException e) {
      e.printStackTrace();
    } catch (ClassCastException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }
  }

  /**
   * Outputs log relative frequencies.
   * 
   * @param filename
   */
  private static void outputStats(String filename) {
    try {
      PrintWriter pw = new PrintWriter(new PrintStream(new FileOutputStream(filename),false,"UTF-8"));
    
      double logTotal = Math.log(counter.totalCount());
      assert logTotal > 0;
      for (String key : counter.keySet()) {
        double logFreq = Math.log(counter.getCount(key)) - logTotal;
        pw.printf("%s\t%f%n", key, logFreq);
      }
      pw.close();
      
    } catch (UnsupportedEncodingException e) {
      e.printStackTrace();
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }

  /**
   * @param args
   */
  public static void main(String[] args) {
    if (args.length != 2) {
      System.err.printf("Usage: java %s [-w|-t|-wt] filename%n", IndexAnnotationGrossStatistics.class.getName());
      System.exit(-1);
    }
  
    String option = args[0];
    String outfileName = args[1];
  
    if (option.equals("-w")) {
      statsFromIndex(AnnotationType.Word);
      
    } else if (option.equals("-t")) {
      statsFromIndex(AnnotationType.Tag);
      
    } else {
      System.err.printf("Usage: java %s [-w|-t|-wt] filename%n", IndexAnnotationGrossStatistics.class.getName());
      System.exit(-1);
    }

    outputStats(outfileName);
    
    System.out.println("Done!");
  }  
}
