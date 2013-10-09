package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.io.UnsupportedEncodingException;
import java.util.Stack;
import java.util.zip.GZIPInputStream;

import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;

/**
 * Defines quite a few utility functions for processing XML documents
 * using SAX.
 * <br>
 * One can override getHandler to provide a different kind of document
 * handler which handles the documents in a specific way, and one
 * can override finishXML to finish the processed XML as needed.
 *
 * @author John Bauer
 */
public class KBPFileProcessor {
  final SAXParser parser;

  public int filesProcessed = 0;

  public KBPFileProcessor() { 
    try {
      parser = SAXParserFactory.newInstance().newSAXParser();
    } catch (ParserConfigurationException e) {
      throw new RuntimeException(e);
    } catch (SAXException e) {
      throw new RuntimeException(e);
    }
  }


  /**
   * Reads the given filename (assumed to be a file at this point)
   * by creating a reader and then processing that reader with 
   * processXML(Reader)
   */
  public void processXML(String filename) {
    try {
      File file = new File(filename);
      FileInputStream fis = new FileInputStream(file);
      InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
      processXML(new BufferedReader(isr), filename);
    } catch (FileNotFoundException e) {
      throw new RuntimeException(e);
    } catch (UnsupportedEncodingException e) {
      throw new RuntimeException(e);
    } catch (SAXException e) {
      throw new RuntimeException("Error while parsing " + filename + 
                                 ":\n" + e.toString(), e);
    }
  }


  /**
   * Feeds the given reader to the sax parser using the handler
   * created in the constructor.
   * <br>
   * TODO: throw a different kind of exception?
   */
  public void processXML(Reader input, String filename) 
    throws SAXException
  {
    try {
      InputSource source = new InputSource(input);
      source.setEncoding("UTF-8");

      DefaultHandler handler = getHandler();
      parser.parse(source, handler);
      finishXML(handler, filename);
      
      ++filesProcessed;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * This method is factored out so subclasses can override it as necessary
   */
  public void processXMLText(String xml, String filename) 
    throws SAXException
  {
    processXML(new StringReader(xml), filename);
  }

  public void processTarArchive(String filename, boolean removeDoctype) {
    try {
      InputStream is = new BufferedInputStream(new FileInputStream(filename));
      if (filename.endsWith(".gz") || filename.endsWith(".tgz")) {
        is = new GZIPInputStream(is);
      }
      // Create a TarInputStream
      TarArchiveInputStream tis = new TarArchiveInputStream(is);
      TarArchiveEntry entry;
      // we will use this array to make sure files terminate when expected
      byte spillover[] = new byte[100];
      while ((entry = tis.getNextTarEntry()) != null) {
        if (entry.getSize() <= 0) {
          // Might be a directory entry, for example
          //System.out.println("Skipping " + entry.getName());
          continue;
        }
        // TODO: not sure the difference between getSize and getRealSize
        byte data[] = new byte[(int) entry.getSize()];
        int count = tis.read(data);
        if (count != entry.getSize() || tis.read(spillover) >= 0) {
          String error;
          if (count == entry.getSize()) {
            error = "Read more than the expected " + entry.getSize();
          } else {
            error = "Read " + count + "; expected " + entry.getSize();
          }
          throw new RuntimeException("Data length and given size for " +
                                     filename + ":" + entry.getName() +
                                     " did not match.  " + error);
        }
        // TODO: pay attention to encoding where needed
        String xml = new String(data);
        if (removeDoctype) {
          xml = xml.replaceAll("<!DOCTYPE[^>]*>", "");
        }
        
        processXMLText(xml, entry.getName());
      }
      
      tis.close();
      is.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    } catch (SAXException e) {
      throw new RuntimeException(e);
    }
  }

  protected DefaultHandler getHandler() {
    return new DefaultHandler();
  }

  protected void finishXML(DefaultHandler handler, String filename) 
    throws SAXException {}

  /**
   * Recursively index all of the files in filenames.  If a filename
   * points to a directory, recurse, otherwise count that file.
   * <br>
   * TODO: this is exactly the same as CountNestedFiles.recursiveCount.  
   * Refactor.
   */
  public void recursiveProcess(String[] filenames) {
    for (String filename : filenames) {
      File file = new File(filename);
      if (file.isDirectory()) {
        String[] sublist = file.list();
        for (int i = 0; i < sublist.length; ++i) {
          sublist[i] = filename + File.separator + sublist[i];
        }
        recursiveProcess(sublist);
      } else {
        processXML(filename);
        if (filesProcessed % 1000 == 0)
          System.out.println("Counted " + filesProcessed + " files");
      }
    }
  }


  /**
   * This exception is used to communicate that there was an
   * unexpected recurrence of the set of tags we're looking at.  In
   * general, the tags we parse for the Lucene index over the KBP only
   * occur once in each of the documents.  If there is a duplicate, we
   * throw this kind of exception to make it clear what the problem
   * is.  It keeps track of the set of tags we saw a second time for
   * easy reporting of the error.
   */
  public static class DuplicatedTagException extends SAXException {
    /**
     * 
     */
    private static final long serialVersionUID = 1L;
    final Stack<String> tagStack;

    DuplicatedTagException(Stack<String> tagStack) {
      this.tagStack = tagStack;
    }

    @Override 
    public String getMessage() {
      return "Duplicated tags: " + tagStack + "\n" + super.getMessage();
    }

    @Override 
    public String toString() {
      return "Duplicated tags: " + tagStack + "\n" + super.toString();
    }
  }
}