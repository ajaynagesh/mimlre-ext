package edu.stanford.nlp.kbp.slotfilling.index;

/*
 * Reader for TAC_2009_KBP_Evaluation_Reference_Knowledge_Base files
 */

import java.io.File;
import java.io.IOException;
import java.util.logging.Logger;

import javax.xml.parsers.ParserConfigurationException;

import edu.stanford.nlp.kbp.slotfilling.common.DomReader;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import edu.stanford.nlp.io.IOUtils;

abstract public class SimpleKBPReader extends DomReader {
  private final Logger logger;
  
  private int numFilesRead = 0;
  private int numEntitiesRead = 0;
    
  public SimpleKBPReader() {
    logger = Logger.getLogger(SimpleKBPReader.class.getName());
  }
  
  public void read(String kbEvalRefDir) throws IOException, SAXException, ParserConfigurationException {
    logger.info("Reading from directory " + kbEvalRefDir);
    for (File file : IOUtils.iterFilesRecursive(new File(kbEvalRefDir), "xml")) {
      readFile(file);
    }

    logger.info("Read " + numFilesRead + " files and " + numEntitiesRead + " entities)");
  }

  private void readFile(File file) throws IOException, SAXException,
  ParserConfigurationException {
    logger.info("Reading " + file.getName() + " (read " + numFilesRead
        + " files and " + numEntitiesRead + " entities so far)");
    Document document = readDocument(file);

    NodeList entities = document.getElementsByTagName("entity");
    for (int i = 0; i < entities.getLength(); i++) {
      Node entity = entities.item(i);
      readEntityNode(entity);
    }
    
    numFilesRead++;
  }

  private void readEntityNode(Node entity) throws IOException {
    String entityNerType = getAttributeValue(entity, "type");
    
    // skip non-PER/ORG entities
    if (!entityNerType.equals("PER") && !entityNerType.equals("ORG")) {
      return;
    }
    
    String entityName = getAttributeValue(entity, "name");
    String entityId = getAttributeValue(entity, "id");
    String entityWikiText = getChildByName(entity, "wiki_text").getChildNodes().item(0).getNodeValue();
    gotEntity(entityName, entityNerType, entityId, entityWikiText);
    
    numEntitiesRead++;
  }

  /**
   * Subclasses must implement this method to receive the entities which have
   * been read from the KBP_Evaluation Reference Knowledge Base.
   */
  abstract public void gotEntity(String name, String nerType, String id, String wikiText);
}