package edu.stanford.nlp.kbp.slotfilling.distantsupervision;

import edu.stanford.nlp.kbp.slotfilling.common.EntityType;
import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.index.KbpXmlHandler;

import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.Attributes;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

public class TaskXMLParser extends KbpXmlHandler {
  static final String[] NEW_QUERY_TAGS = {"kbpslotfill", "query"};
  static final String[] NAME_TAGS = {"kbpslotfill", "query", "name"};
  static final String[] DOCID_TAGS = {"kbpslotfill", "query", "docid"};
  // treebeard, leaflock, beechbone, etc
  static final String[] ENTTYPE_TAGS = {"kbpslotfill", "query", "enttype"};
  static final String[] NODEID_TAGS = {"kbpslotfill", "query", "nodeid"};
  static final String[] IGNORE_TAGS = {"kbpslotfill", "query", "ignore"};

  static final String ID_ATTRIBUTE = "id";
  
  //sometimes the query file has node id as nil. Assign them nil+nilIDCounter;
  static int nilIDCounter = 0;

  /**
   * Returns a list of the EntityMentions contained in the Reader passed in.
   * <br>
   * This can throw exceptions in the following circumstances:
   * <br>
   * If there is a nested &lt;query&gt; tag, it will throw a SAXException
   * <br>
   * If there is a &lt;query&gt; tag with no id attribute, it will also throw
   * a SAXException
   * <br>
   * If any of the name, enttype, or nodeid fields are missing, it once
   * again throws a SAXException
   * <br>
   * If there is a problem with the reader passed in, it may throw an
   * IOException
   */
  public static List<KBPEntity> parseQueryFile(Reader input) 
    throws IOException, SAXException
  {
    InputSource source = new InputSource(input);
    source.setEncoding("UTF-8");
    
    TaskXMLParser handler = new TaskXMLParser();
    
    try {
      SAXParser parser = SAXParserFactory.newInstance().newSAXParser();
      parser.parse(source, handler);
    } catch(ParserConfigurationException e) {
      throw new RuntimeException(e);
    }
    return handler.mentions;
  }

  public static List<KBPEntity> parseQueryFile(String filename)
    throws IOException, SAXException
  {
    BufferedReader reader = new BufferedReader(new FileReader(filename));
    List<KBPEntity> mentions = parseQueryFile(reader);
    reader.close();
    return mentions;
  }

  /**
   * The only way to use one of these objects is through the
   * parseQueryFile method
   */
  private TaskXMLParser() {}

  List<KBPEntity> mentions = new ArrayList<KBPEntity>();

  KBPEntity currentMention = null;  
  StringBuilder currentText = null;

  @Override
  public void startElement(String uri, String localName, 
                           String qName, Attributes attributes)
    throws SAXException
  {
    super.startElement(uri, localName, qName, attributes);

    if (matchesTags(NEW_QUERY_TAGS)) {
      if (currentMention != null)
        throw new RuntimeException("Unexpected nested query after query #" + 
                                   mentions.size());
      currentMention = new KBPEntity();
      String id = attributes.getValue(ID_ATTRIBUTE);
      Log.fine("Query ID is " + id);
      if (id == null) 
        throw new SAXException("Query #" + (mentions.size() + 1) + 
                               " has no id, " +
                               "what are we supposed to do with that?");
      currentMention.queryId = id;
    } else if (matchesTags(NAME_TAGS) || matchesTags(DOCID_TAGS) ||
               matchesTags(ENTTYPE_TAGS) || matchesTags(NODEID_TAGS) ||
               matchesTags(IGNORE_TAGS)) {
      currentText = new StringBuilder();
    }
  }
  
  @Override
  public void endElement(String uri, String localName, 
                         String qName) 
    throws SAXException
  {
    if (currentText != null) {
      String text = currentText.toString().trim();
      if (matchesTags(NAME_TAGS)) {
        currentMention.name = text;
      } else if (matchesTags(DOCID_TAGS)) {
        currentMention.docid = text;
      } else if (matchesTags(ENTTYPE_TAGS)) {
        currentMention.type = EntityType.fromXmlRepresentation(text);
      } else if (matchesTags(NODEID_TAGS)) {
        currentMention.id = text;
      } else if (matchesTags(IGNORE_TAGS)) {
        if (!text.equals("")) {
          String[] ignorables = text.split("\\s+");
          Set<String> ignoredSlots = new HashSet<String>();
          for (String ignore : ignorables) {
            ignoredSlots.add(ignore);
          }
          currentMention.ignoredSlots = ignoredSlots;
        }
      } else {
        throw new RuntimeException("Programmer error!  " + 
                                   "Tags handled in startElement are not " +
                                   "handled in endElement");
      }
      currentText = null;
    }
    if (matchesTags(NEW_QUERY_TAGS)) {
      boolean shouldAdd = true;
      if (currentMention == null) {
        throw new NullPointerException("Somehow exited a query block with " +
                                       "currentMention set to null");
      }
      if (currentMention.ignoredSlots == null) {
        currentMention.ignoredSlots = Collections.emptySet();
      }
      if (currentMention.type == null) {
        System.err.println("Query #" + (mentions.size() + 1) +
                           " has no known type. It was probably GPE. Skipping...");
        shouldAdd = false;
      } 
      if (currentMention.name == null) {
        throw new SAXException("Query #" + (mentions.size() + 1) +
                               " has no name");
      } 
      if (currentMention.id == null) {
        throw new SAXException("Query #" + (mentions.size() + 1) +
                               " has no nodeid");
      } 
      if (currentMention.queryId == null) {
        throw new SAXException("Query #" + (mentions.size() + 1) +
                               " has no queryid");
      }
      if(currentMention.id.equals("NIL"))
      {
        String newId = "NIL"+nilIDCounter;
        Log.info("Query # " + currentMention.queryId + " has id as NIL. Assigning it random id " + nilIDCounter);
        currentMention.id = newId;
        nilIDCounter ++;
      }
      if(shouldAdd) mentions.add(currentMention);
      currentMention = null;
    }

    super.endElement(uri, localName, qName);
  }

  /**
   * If we're in a set of tags where we care about the text, save
   * the text.  If we're in a set of tags where we remove the
   * underscores, do that first.
   */
  @Override
  public void characters(char buf[], int offset, int len) {
    if (currentText != null) {
      currentText.append(new String(buf, offset, len));
    }
  }
  
}
