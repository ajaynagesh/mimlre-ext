package edu.stanford.nlp.kbp.slotfilling.index;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.util.Version;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.SimpleFSDirectory;

import java.io.File;
import java.io.IOException;

import java.util.EnumMap;
import java.util.Map;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class BuildLucene extends KBPFileProcessor {
  final IndexWriter writer;
  
  public BuildLucene(String indexDir) 
    throws IOException
  {
    this(new SimpleFSDirectory(new File(indexDir)));
  }

  @SuppressWarnings("deprecation")
  public BuildLucene(Directory directory) 
    throws IOException
  {
    super();

    writer = new IndexWriter(directory,
                             new StandardAnalyzer(Version.LUCENE_CURRENT), 
                             true, IndexWriter.MaxFieldLength.LIMITED);
  }

  /**
   * Optimizes and then writes out the index.
   */
  public void finishIndex() {
    try {
      System.out.println("Optimizing");
      writer.optimize();
      System.out.println("Closing");
      writer.close();
      System.out.println("Done!  Total files indexed: " + filesProcessed);
    } catch (IOException e) {
      throw new RuntimeException(e);
    } 
  }

  
  @Override
  protected DefaultHandler getHandler() {
    return new LuceneDocumentHandler();
  }

  @Override
  protected void finishXML(DefaultHandler handler, String filename) {
    try {
      if (handler instanceof LuceneDocumentHandler) {
        writer.addDocument(((LuceneDocumentHandler) handler).doc);
      } else {
        throw new IllegalArgumentException("Expected a LuceneDocumentHandler");
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * To run: java &lt;program name&gt; [filename...]
   */
  static public void main(String[] args) 
    throws IOException
  {
    if (args.length < 2) {
      System.out.println("Arg 1 must be the location to put the index, " +
                         "and args 2+ must be files or directories to index.");
      System.exit(2);
    }

    File file = new File(args[0]);
    if (file.exists() && !file.isDirectory()) {
      System.out.println("Hey, be careful, you almost overwrote an " +
                         "important file: " + args[0]);
      System.exit(1);
    } else if (file.exists()) {
      System.out.println("There's already something here: " + args[0]);
      System.out.println("Try running with a directory that doesn't " +
                         "already exist; the directory will be created");
      System.exit(1);
    }

    BuildLucene indexer = new BuildLucene(args[0]);
    String[] newargs = new String[args.length - 1];
    for (int i = 0; i < args.length - 1; ++i) {
      newargs[i] = args[i + 1];
    }
    indexer.recursiveProcess(newargs);
    indexer.finishIndex();
  }
  
  /**
   * This handler takes a document formatted as KBP does and extracts
   * the datetime, docid, text, and headline text.
   */
  public static class LuceneDocumentHandler extends KbpXmlHandler {
    Map<KBPField, String> fields = new EnumMap<KBPField, String>(KBPField.class);

    boolean removeUnderscores = false;
    StringBuilder currentText = null;

    final Document doc = new Document();

    public Document getDocument()
    {
      return doc;
    }

    /**
     * Helper method: add a field to the Document we're creating
     */
    private void addField(Document doc, KBPField field) {
      String data = fields.get(field);
      if (data != null) {
        doc.add(new Field(field.fieldName(), data, Field.Store.YES, 
                          field.indexingStrategy()));
      }
    }

    @Override
    public void endDocument() 
      throws SAXException
    {
      //System.out.println("Docid: '" + docid + "'");
      //System.out.println(" Datetime: '" + datetime + "'");
      //System.out.println(" Headline: '" + headline + "'");
      //System.out.println(" Text: " + text);

      addField(doc, KBPField.DATETIME);
      addField(doc, KBPField.DOCID);
      addField(doc, KBPField.HEADLINE);
      addField(doc, KBPField.TEXT);
    }

    /**
     * If we're already in a set of tags where we care about the
     * insides, then save the new tag.  Otherwise, check to see if we
     * care about the insides.
     */
    @Override
    public void startElement(String uri, String localName, 
                             String qName, Attributes attributes) 
    {
      String name = ((!localName.equals("")) ? localName : qName);
      tagStack.push(name);

      //System.out.println(tagStack);

      if (currentText != null) {
        currentText.append("<");
        currentText.append(name);
        currentText.append(">");
      } else if (matchesTags(DATETIME_TAGS) || matchesTags(DOCID_TAGS) ||
                 matchesTags(TEXT_TAGS) || matchesTags(HEADLINE_TAGS)) {
        currentText = new StringBuilder();
      }

      if (currentText != null && matchesTags(SPEAKER_TAGS)) {
        removeUnderscores = true;
      }
    }

    private void foundField(KBPField field) 
      throws SAXException
    {
      if (fields.containsKey(field))
        throw new DuplicatedTagException(tagStack);
      fields.put(field, currentText.toString());
      currentText = null;
    }

    /**
     * If we're in a set of tags where we care about the insides,
     * check to see if we're now leaving the set of tags we cared
     * about.  If we are, save the text and clear the StringBuilder.
     * Otherwise, just add the current close tag to the text and keep
     * going.
     */
    @Override
    public void endElement(String uri, String localName, 
                           String qName) 
      throws SAXException
    {
      if (currentText != null) {
        // if we were in the correct set of tags, save the current text
        // Otherwise, if the current text wasn't null, append the tag
        String name = ((!localName.equals("")) ? localName : qName);
        if (matchesTags(DATETIME_TAGS)) {
          foundField(KBPField.DATETIME);
        } else if (matchesTags(DOCID_TAGS)) {
          foundField(KBPField.DOCID);
        } else if (matchesTags(TEXT_TAGS)) {
          foundField(KBPField.TEXT);
        } else if (matchesTags(HEADLINE_TAGS)) {
          foundField(KBPField.HEADLINE);
        } else {
          currentText.append("</");
          currentText.append(name);
          currentText.append(">");
        }

        if (removeUnderscores && matchesTags(SPEAKER_TAGS)) {
          removeUnderscores = false;
        }
      }
      tagStack.pop();
    }

    /**
     * If we're in a set of tags where we care about the text, save
     * the text.  If we're in a set of tags where we remove the
     * underscores, do that first.
     */
    @Override
    public void characters(char buf[], int offset, int len) {
      if (currentText != null) {
        String newText = new String(buf, offset, len);
        if (removeUnderscores) {
          newText = newText.replaceAll("_", " ");
        }
        currentText.append(newText);
      }
    }
  }
}