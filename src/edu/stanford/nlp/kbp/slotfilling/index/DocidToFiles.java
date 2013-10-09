package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.*;

import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Pattern;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class DocidToFiles extends KBPFileProcessor {
  Map<String, String> docidToFileMap = new TreeMap<String, String>();
  Map<String, String> fileToDocidMap = new TreeMap<String, String>();

  @Override
  protected DefaultHandler getHandler() {
    return new DocidToFileHandler();
  }
  
  public Map<String, String> getDocidToFileMap() { return docidToFileMap; }
  public Map<String, String> getFileToDocidMap() { return fileToDocidMap; }

  @Override
  protected void finishXML(DefaultHandler handler, String filename) 
    throws SAXException
  {
    String docid = null;
    if (handler instanceof DocidToFileHandler) {
      docid = ((DocidToFileHandler) handler).docid.trim();
    } else {
      throw new IllegalArgumentException("Expected DocidToFileHandler");
    }

    if (docid == null) {
      throw new SAXException("File " + filename + "had no docid");
    }

    if (docidToFileMap.containsKey(docid)) {
      String error = ("Saw docid " + docid + " twice: once in " + filename +
                      " and once in " + docidToFileMap.get(docid));
      System.out.println(error);
      return;
      //throw new IllegalArgumentException(error);
    }

    if (fileToDocidMap.containsKey(filename)) {
      String error = ("Saw filename " + filename + " twice: once in " + docid +
                      " and once in " + fileToDocidMap.get(filename));
      System.out.println(error);
      return;
      //throw new IllegalArgumentException("Saw file " + filename + " twice");
    }

    docidToFileMap.put(docid, filename);
    fileToDocidMap.put(filename, docid);
  }

  public void writeMapping(Writer output)
    throws IOException
  {
    for (Map.Entry<String, String> mapping : docidToFileMap.entrySet()) {
      output.write(mapping.getKey() + " : " + mapping.getValue() + "\n");
    }
    output.flush();
  }

  private static Pattern delimiterPattern = Pattern.compile(":");
  public void readMapping(Reader input)
    throws IOException
  {
    BufferedReader br = (input instanceof BufferedReader)?
            (BufferedReader) input: new BufferedReader(input);
    String line;
    while ((line = br.readLine()) != null) {
      String[] fields = delimiterPattern.split(line);
      if (fields.length == 2) {
          String docid = fields[0].trim();
          String filename = fields[1].trim();
          docidToFileMap.put(docid, filename);
          fileToDocidMap.put(filename, docid);
      } else {
          System.err.println("WARNING: Unexpected number of fields " + fields.length +
                  " while reading docid to file mapping " + line);
      }
    }
  }

  public String getFile(String docid)
  {
    return docidToFileMap.get(docid);
  }

  public String getDocid(String filename)
  {
    return fileToDocidMap.get(filename);
  }

  /**
   * To run: java &lt;program name&gt; [filename...]
   */
  static public void main(String[] args) 
    throws IOException
  {
    DocidToFiles docids = new DocidToFiles();
    docids.recursiveProcess(args);
    docids.writeMapping(new BufferedWriter(new OutputStreamWriter(System.out)));
  }
  
  class DocidToFileHandler extends KbpXmlHandler {
    StringBuilder docidBuilder = null;
    String docid = null;

    @Override
    public void startElement(String uri, String localName, 
                             String qName, Attributes attributes) 
      throws SAXException
    {
      String name = ((!localName.equals("")) ? localName : qName);
      tagStack.push(name);

      if (docidBuilder != null) {
        throw new SAXException("Already inside <docid>, " +
                               "but found another nested element");
      }

      if (matchesTags(DOCID_TAGS)) {
        if (docid != null) {
          throw new DuplicatedTagException(tagStack);
        }

        docidBuilder = new StringBuilder();
      }
    }

    @Override
    public void endElement(String uri, String localName, String qName) 
    {
      tagStack.pop();
      if (docidBuilder != null) {
        docid = docidBuilder.toString();
        docidBuilder = null;
      }
    }

    @Override
    public void characters(char buf[], int offset, int len) {
      if (docidBuilder != null) {
        String newText = new String(buf, offset, len);
        docidBuilder.append(newText);
      }
    }
  }
}