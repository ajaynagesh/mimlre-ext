package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.logging.Level;
import java.util.zip.GZIPInputStream;

import javax.xml.parsers.ParserConfigurationException;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.SimpleFSDirectory;
import org.apache.lucene.util.Version;

import org.xml.sax.SAXException;

import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.KBPSlot;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.KBPDomReader;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.TaskXMLParser;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationSerializer;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.MutableInteger;
import edu.stanford.nlp.util.StringUtils;

/**
 * This class takes an existing Lucene index and writes all the
 * documents in the index out to the pipeline.
 * @author John Bauer
 * Sample command line:
 * <br>
 *  nohup time java -mx8g edu.stanford.nlp.kbp.index.LucenePipelineCacher 
 *   -props ~/codebase/javanlp/projects/research/src/edu/stanford/nlp/kbp/kbp.properties 
 *   -cacher.input /u/nlp/data/TAC-KBP2010/TAC_2010_KBP_Source_Data_Index 
 *   -cacher.output ~/filtered -cacher.shard 10 -cacher.numshards 100 
 *   -cacher.filter /u/nlp/data/TAC-KBP2010/TAC_2009_KBP_Evaluation_Slot_Filling_List/data/slot_filling_query_list.xml 
 *      > ~/split_filtered.txt 2>&1 &
 * <br>
 *  nohup time java -mx8g edu.stanford.nlp.kbp.index.LucenePipelineCacher 
 *    -props ~/codebase/javanlp/projects/research/src/edu/stanford/nlp/kbp/kbp.properties 
 *    -cacher.input /u/nlp/data/TAC-KBP2010/TAC_2010_KBP_Source_Data_Index 
 *    -cacher.output /scr/horatio/split_8 -cacher.shard 8 -cacher.numshards 20
 *    -cacher.filter "/u/nlp/data/TAC-KBP2010/TAC_2009_KBP_Evaluation_Slot_Filling_List/data/slot_filling_query_list.xml;/u/nlp/data/TAC-KBP2010/test2010/TAC_2010_KBP_Evaluation_Slot_Filling_Queries/data/tac_2010_kbp_evaluation_slot_filling_queries.xml;/u/nlp/data/TAC-KBP2010/participant_annotation_queries/combined.xml" 
 *    -cacher.skip 1213148 > ~/split_8.txt 2>&1 &
 */
public class LucenePipelineCacher {
  static final String SHARD_PROPERTY = "cacher.shard";
  static final String SHARD_DEFAULT = "-1";
  static final String NUM_SHARDS_PROPERTY = "cacher.numshards";
  static final String NUM_SHARDS_DEFAULT = "-1";
  static final String IS_WIKIPEDIA_PROPERTY = "cacher.wikipedia";
  static final String IS_WIKIPEDIA_DEFAULT = "false";
  static final String INPUT_PROPERTY = "cacher.input";
  static final String OUTPUT_PROPERTY = "cacher.output";  
  static final String FILTER_PROPERTY = "cacher.filter";
  static final String SKIP_IDS_PROPERTY = "cacher.skip";

  StanfordCoreNLP pipeline = null;

  public LucenePipelineCacher(Properties pipelineProperties) {
    pipeline = new StanfordCoreNLP(pipelineProperties);
  }

  public static List<String> recursiveFindFiles(String pattern,
                                                String ... startingLocations) {
    ArrayList<String> results = new ArrayList<String>();
    for (String filename : startingLocations) {
      File file = new File(filename);
      if (file.isDirectory()) {
        String[] sublist = file.list();
        for (int i = 0; i < sublist.length; ++i) {
          sublist[i] = filename + File.separator + sublist[i];
        }
        results.addAll(recursiveFindFiles(pattern, sublist));
      } else {
        String path = file.getAbsolutePath();
        if (path.matches(pattern))
          results.add(path);
      }
    }
    return results;
  }

  public static Set<Integer> findInterestingDocuments(KBPDomReader domReader,
                                                      String inputDirectory,
                                                      String queryFileString)
    throws IOException, SAXException, ParserConfigurationException
  {
    List<String> files = recursiveFindFiles(".*\\.xml", 
                                            queryFileString.split(";"));
    return  findInterestingDocuments(domReader, inputDirectory, files);
  }

  public static Set<Integer> findInterestingDocuments(KBPDomReader domReader,
                                                      String inputDirectory,
                                                      List<String> queryFiles) 
    throws IOException, SAXException, ParserConfigurationException
  {
    List<KBPEntity> task = new ArrayList<KBPEntity>();
    for (String queryFile : queryFiles) {
      Collection<KBPEntity> newtasks = null;
      newtasks = TaskXMLParser.parseQueryFile(queryFile);
      if (newtasks.size() == 0) {
        Log.severe("File " + queryFile + " is not a query file; " +
                           "attempting to read as a KB file");
        Map<KBPEntity, List<KBPSlot>> newKB = domReader.parse(queryFile);
        newtasks = newKB.keySet();
      }

      Log.severe("Found " + newtasks.size() + " tasks for " + 
                         queryFile);
      task.addAll(newtasks);
    }
    Directory input = new SimpleFSDirectory(new File(inputDirectory));
    return findInterestingDocuments(input, task);
  }

  public static Set<Integer> 
    findInterestingDocuments(Directory inputDirectory,
                             List<KBPEntity> task) 
    throws IOException
  {
    Set<String> stopWords = QueryUtils.standardStopWords();

    IndexSearcher searcher = new IndexSearcher(inputDirectory);
    int numDocs = searcher.maxDoc();

    Set<Integer> interestingDocuments = new HashSet<Integer>();

    for (KBPEntity entity : task) {
      String name = QueryUtils.rewriteQueryTerm(entity.name);
      TopDocs results;
      try {
        Query query = QueryUtils.buildPhraseQuery(name, 
                                                  KBPField.TEXT.fieldName(), 
                                                  stopWords);
        results = searcher.search(query, numDocs);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }

      List<Integer> docids = new ArrayList<Integer>();
      for (ScoreDoc scoreDoc : results.scoreDocs) {
        int docId = scoreDoc.doc;
        docids.add(docId);
      }
      interestingDocuments.addAll(docids);
    }
    return interestingDocuments;
  }

  public void runPipeline(String inputDirectory,
                          String outputDirectory,
                          Set<Integer> interestingDocs,
                          Set<Integer> docsToSkip,
                          int shardNum, int numShards,
                          boolean isWikipedia) 
    throws IOException
  {
    runPipeline(new SimpleFSDirectory(new File(inputDirectory)),
                new SimpleFSDirectory(new File(outputDirectory)),
                interestingDocs, docsToSkip, 
                shardNum, numShards, isWikipedia);
  }

  @SuppressWarnings("deprecation")
  public void runPipeline(Directory inputDirectory,
                          Directory outputDirectory,
                          Set<Integer> interestingDocs,
                          Set<Integer> docsToSkip,
                          int shardNum, int numShards,
                          boolean isWikipedia) 
    throws IOException
  {
    IndexSearcher searcher = new IndexSearcher(inputDirectory);
    int numDocs = searcher.maxDoc();
    MutableInteger serSize = new MutableInteger(0);

    int toProcess = 0;
    if (numShards <= 0) {
      toProcess = numDocs;
    } else {
      toProcess = (numDocs / numShards + 
                   (numDocs % numShards > shardNum ? 1 : 0));
    }
    Log.severe("Processing " + toProcess + " of " + numDocs + " docs");
    
    IndexWriter writer = 
      new IndexWriter(outputDirectory,
                      new StandardAnalyzer(Version.LUCENE_CURRENT), 
                      true, IndexWriter.MaxFieldLength.LIMITED);
    int processed = 0;
    int annotated = 0;
    for (int i = 0; i < numDocs; ++i) {
      if (numShards > 0 && i % numShards != shardNum)
        continue;

      Log.severe("Processing document at position #" + i);

      Document doc = searcher.doc(i);
      String docid = doc.get(KBPField.DOCID.fieldName());
      Log.severe("Parsing document with docid " + docid);
      System.err.println("Parsing document with docid " + docid);

      // if the doc id we are looking at is one we care about, or we
      // didn't specify which docs we care about...
      if (interestingDocs != null && !interestingDocs.contains(i)) {
        Log.severe("  Not interesting; will not be saved in the new index!");
      } else if (docsToSkip != null && docsToSkip.contains(i)) {
        Log.severe("  Interesting but you said to skip it; " +
                           "will not be saved in the new index!");
      } else {
        // then add the Annotation for this doc as a serialized object
        String text = IndexExtractor.extractText(doc, isWikipedia);
        Annotation annotatedText = new Annotation(text);
        pipeline.annotate(annotatedText);

        addAnnotationToDoc(doc, annotatedText, serSize);
        ++annotated;
        
        if(isWikipedia){
          // the new (2010) wikipedia index has "title" and "content" but no "docid" and "text"
          // copy title to docid; move content to text
          // the above operations make the regenerated Wikipedia index compatible with other indices we have
          copyField(doc, KBPField.WIKITITLE, KBPField.DOCID, false);
          copyField(doc, KBPField.WIKICONTENT, KBPField.TEXT, true);
        }
        
        // save in the new index
        writer.addDocument(doc);
        Log.severe("Completed document with docid " + docid);
        System.err.println("Completed document with docid " + docid);
      }
      
      ++processed;
      if (processed % 100 == 0 || processed == toProcess) {
        Log.severe("Processed " + processed + " docs; annotated " +
                           annotated + " docs.");
      }
    }

    Log.severe("Finishing index...");
    writer.optimize();
    writer.close();
    Log.severe("Done!");
    Log.severe("Annotated " + annotated 
        + " documents. Annotation size = " + serSize.intValue());
  }
  
  public static boolean copyField(Document doc, KBPField src, KBPField dst, boolean removeSrc) {
    // do not copy if destination exists
    String dstText = doc.get(dst.fieldName());
    if(dstText != null) return false;
    
    // remove the old field (just in case), otherwise we can't add the new one
    doc.removeFields(dst.fieldName()); 
    
    // copy src to dst
    String srcText = doc.get(src.fieldName());
    doc.add(new Field(dst.fieldName(), 
        srcText, Field.Store.YES, 
        dst.indexingStrategy()));
    
    // remove src if requested
    if(removeSrc){
      doc.removeFields(src.fieldName()); 
    }
    
    return true;
  }
  
  private static final boolean COMPRESS_ANNOTATIONS = true;

  /**
   * Produces a serialized object from the annotation and attaches it
   * to the Document, using the COREMAP field.
   */
  public static void addAnnotationToDoc(Document doc, 
                                        Annotation annotation,
                                        MutableInteger serSize)
    throws IOException
  {
    // remove the old annotation, otherwise we can't add the new one
    doc.removeFields(KBPField.COREMAP.fieldName());
    
    //
    // serialize everything we care of
    //
    ByteArrayOutputStream byteOutput = new ByteArrayOutputStream();
    AnnotationSerializer ser = 
      new KBPAnnotationSerializer(COMPRESS_ANNOTATIONS, true);
    ser.save(annotation, byteOutput);
    byteOutput.close();
    if(serSize != null) serSize.incValue(byteOutput.size());

    String annotationAsString = byteOutput.toString("ISO-8859-1");
    // Log.severe("BEGIN ANNOTATION\n" + annotationAsString + "END ANNOTATION");
    doc.add(new Field(KBPField.COREMAP.fieldName(), 
                      annotationAsString, Field.Store.YES, 
                      KBPField.COREMAP.indexingStrategy()));    
  }
  
  public static String getAnnotationStringFromDoc(Document doc) throws UnsupportedEncodingException, IOException {
    String annotationBytes = doc.get(KBPField.COREMAP.fieldName());    
    BufferedReader is = new BufferedReader(new InputStreamReader(new GZIPInputStream(new BufferedInputStream(new ByteArrayInputStream(annotationBytes.getBytes("ISO-8859-1"))))));
    StringBuffer os = new StringBuffer();
    String line;
    while((line = is.readLine()) != null){
      os.append(line);
      os.append("\n");
    }
    is.close();
    return os.toString();
  }
  
  public static Annotation getAnnotationFromDoc(Document doc) 
    throws IOException, ClassNotFoundException {
    return getAnnotationFromDoc(doc, null);
  }

  public static Annotation getAnnotationFromDocNoExceptions(Document doc) {
    try {
      return getAnnotationFromDoc(doc);
    } catch (Exception e) {
      return null;
    }
  }
  
  /**
   * Inverse operation of AddAnnotationToDoc...  returns the
   * Annotation attached to the Document as a serialized object in the
   * COREMAP field.  Throws exceptions if the object stored there
   * isn't actually an Annotation.
   */
  public static Annotation getAnnotationFromDoc(Document doc, StringBuilder sb) 
    throws IOException, ClassNotFoundException
  {
    String annotationBytes = doc.get(KBPField.COREMAP.fieldName());
    if (annotationBytes == null)
      return null;
    if(sb != null){
      if(COMPRESS_ANNOTATIONS){
        BufferedReader is = new BufferedReader(new InputStreamReader(new GZIPInputStream(new ByteArrayInputStream(annotationBytes.getBytes("ISO-8859-1")))));
        String line;
        while((line = is.readLine()) != null){
          sb.append(line);
          sb.append("\n");
        }
        is.close();
      } else {
        sb.append(annotationBytes);
      }
    }
    ByteArrayInputStream byteInput = 
      new ByteArrayInputStream(annotationBytes.getBytes("ISO-8859-1"));
    AnnotationSerializer ser = 
      new KBPAnnotationSerializer(COMPRESS_ANNOTATIONS, true);
    Annotation annotation = ser.load(byteInput);
    return annotation;
  }

  public static void main(String[] args) 
    throws Exception
  {
    Properties properties = StringUtils.argsToProperties(args);   
    Log.setLevel(Level.INFO);

    String inputDirectory = properties.getProperty(INPUT_PROPERTY);
    if (inputDirectory == null) {
      System.err.println("You must set the input directory property, " + 
                         INPUT_PROPERTY);
      System.exit(2);
    }

    String outputDirectory = properties.getProperty(OUTPUT_PROPERTY);
    if (outputDirectory == null) {
      System.err.println("You must set the output directory property, " + 
                         OUTPUT_PROPERTY);
      System.exit(2);
    }

    File file = new File(outputDirectory);
    if (file.exists() && !file.isDirectory()) {
      Log.severe("Hey, be careful, you almost overwrote an " +
                         "important file: " + outputDirectory);
      System.exit(1);
    } else if (file.exists()) {
      Log.severe("There's already something here: " + outputDirectory);
      Log.severe("Try running with a directory that doesn't " +
                         "already exist; the directory will be created");
      System.exit(1);
    }

    boolean isWikipedia = 
      Boolean.valueOf(properties.getProperty(IS_WIKIPEDIA_PROPERTY, 
                                             IS_WIKIPEDIA_DEFAULT));
    int shardNum = 
      Integer.valueOf(properties.getProperty(SHARD_PROPERTY, SHARD_DEFAULT));
    int numShards = 
      Integer.valueOf(properties.getProperty(NUM_SHARDS_PROPERTY, 
                                             NUM_SHARDS_DEFAULT));

    String filter = properties.getProperty(FILTER_PROPERTY);

    Log.severe("Input directory:  " + inputDirectory);
    Log.severe("Output directory: " + outputDirectory);
    Log.severe("Running shard " + shardNum + " of " + numShards);
    Log.severe("Doing wikipedia parsing: " + isWikipedia);
    Log.severe("Filtering with queries from: " + filter);

    Set<Integer> interestingDocuments = null;
    if (filter != null) {
      KBPDomReader domReader = new KBPDomReader(properties);
      interestingDocuments = findInterestingDocuments(domReader,
                                                      inputDirectory,
                                                      filter);
      Log.severe("Found " + interestingDocuments.size() + 
                         " documents to annotate");
    }

    Set<Integer> docsToSkip = null;
    String docsToSkipString = properties.getProperty(SKIP_IDS_PROPERTY);
    if (docsToSkipString != null) {
      docsToSkip = new HashSet<Integer>();
      for (String docIdString : docsToSkipString.split(",")) {
        docsToSkip.add(Integer.valueOf(docIdString));
      }
      Log.severe("Skipping the following documents: " + docsToSkip);
    }
    

    LucenePipelineCacher cacher = new LucenePipelineCacher(properties);
    cacher.runPipeline(inputDirectory, outputDirectory, 
                       interestingDocuments, docsToSkip,
                       shardNum, numShards, isWikipedia);
  }
}
