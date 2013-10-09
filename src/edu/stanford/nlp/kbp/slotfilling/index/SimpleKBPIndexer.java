package edu.stanford.nlp.kbp.slotfilling.index;


import java.io.File;
import java.io.IOException;
import java.util.logging.Logger;

import javax.xml.parsers.ParserConfigurationException;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.SimpleFSDirectory;
import org.apache.lucene.util.Version;
import org.xml.sax.SAXException;

import edu.stanford.nlp.kbp.slotfilling.distantsupervision.KBPDomReader;

/**
 * Builds an index over the text from TAC_2009_KBP_Evaluation_Reference_Knowledge_Base files
 */
public class SimpleKBPIndexer extends SimpleKBPReader {
  private final IndexWriter indexWriter;
  private final Logger logger;
  
  private static boolean doHeuristicCorefReplacement = true;

  public SimpleKBPIndexer(String indexDir) throws IOException, SAXException, ParserConfigurationException {
    this(new SimpleFSDirectory(new File(indexDir)));
  }
  
  @SuppressWarnings("deprecation")
  public SimpleKBPIndexer(Directory indexDir) throws IOException, SAXException, ParserConfigurationException {
    super();
    logger = Logger.getLogger(SimpleKBPIndexer.class.getName());
    
    indexWriter = new IndexWriter(indexDir,
        new StandardAnalyzer(Version.LUCENE_CURRENT), 
        true, IndexWriter.MaxFieldLength.LIMITED);
  }
  
  public void read(String kbEvalRefDir) throws IOException, SAXException, ParserConfigurationException  {
    read(kbEvalRefDir);
    
    logger.info("Optimizing...");
    indexWriter.optimize();
    logger.info("Closing...");
    indexWriter.close();
    logger.info("Done.");
  }

  @Override
  public void gotEntity(String name, String nerType, String id, String wikiText) {
    // if the wiki text starts with the name, take that out
    if (wikiText.startsWith(name)) {
      wikiText = wikiText.substring(name.length());
    }
    // remove parentheses from the name (this ordering is important since text starts with name+parentheses)
    // e.g. "Bull Durham (baseball)\n\nLouis Raphael Durham (born (as Louis Raphael Staub) June 27, 1877 in New Oxford,..."
    name = KBPDomReader.removeParentheses(name);
    
    if (doHeuristicCorefReplacement) {
      wikiText = FastHeuristicCoreference.fastHeuristicCorefReplacement(name, wikiText, nerType.equals("PER"));
    }
    
    Document doc = new Document();
    addField(doc, KBPField.DOCID, id);
    addField(doc, KBPField.HEADLINE, name);
    addField(doc, KBPField.TEXT, wikiText);
    try {
      indexWriter.addDocument(doc);
    } catch (CorruptIndexException e) {
      RuntimeException re = new RuntimeException();
      re.initCause(e);
      throw re;
    } catch (IOException e) {
      RuntimeException re = new RuntimeException();
      re.initCause(e);
      throw re;
    }
  }

  // TODO BuildLucene should switch to this simpler version of addField
  private void addField(Document doc, KBPField field, String text) {
    if (text != null) {
      doc.add(new Field(field.fieldName(), text, Field.Store.YES, field.indexingStrategy()));
    }
  }
  
  public static void main(String [] argv) throws Exception {
    SimpleKBPIndexer indexer = new SimpleKBPIndexer("/tmp/luceneindex");
    indexer.read("/u/nlp/data/TAC-KBP2010/TAC_2009_KBP_Evaluation_Reference_Knowledge_Base/data");
  }
}