package edu.stanford.nlp.kbp.slotfilling.index;


import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.Document;
import org.w3c.dom.Element;

import edu.stanford.nlp.util.XMLUtils;

/**
 * Converts the text from TAC_2009_KBP_Evaluation_Reference_Knowledge_Base files into the same document
 * format as TAC_2010_KBP_Source_Data files.
 */
public class SimpleKBPDocWriter extends SimpleKBPReader {
  private String outputDir;
  
  public SimpleKBPDocWriter(String outputDir) {
    super();
    this.outputDir = outputDir;
  }
  
  @Override
  public void gotEntity(String name, String nerType, String id, String wikiText) {
    File subDir = new File(outputDir, id.substring(0, 4));
    subDir.mkdirs();
    File outputFile = new File(subDir, id + ".xml");
    
    /**
     * Output format looks like this:
         
    <DOC>
    <DOCID> ALHURRA_NEWS13_ARB_20050412_130100-2.LDC2006E92 </DOCID>
    <DOCTYPE SOURCE="broadcast conversation"> STORY </DOCTYPE>
    <DATETIME> 2005-04-12 13:37:04 </DATETIME>
    <BODY>
      ...
    */
    
    Document document = XMLUtils.getXmlParser().newDocument();
    Element docRoot = document.createElement("DOC");
    document.appendChild(docRoot);
    
    Element docId = document.createElement("DOCID");
    docId.setTextContent(id + " " + nerType + " " + name);
    docRoot.appendChild(docId);

    Element docType = document.createElement("DOCTYPE");
    docType.setTextContent("WEB TEXT");
    docType.setAttribute("SOURCE", "wikipedia");
    docRoot.appendChild(docType);

    Element dateTime = document.createElement("DATETIME");
    docRoot.appendChild(dateTime);

    Element body = document.createElement("BODY");
    body.setTextContent(wikiText);
    docRoot.appendChild(body);

    try {
      TransformerFactory transformerFactory = TransformerFactory.newInstance();
      Transformer transformer = transformerFactory.newTransformer();
      transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");
      transformer.setOutputProperty(OutputKeys.INDENT, "yes");

      FileWriter fw = new FileWriter(outputFile);
      StreamResult result = new StreamResult(fw);
      DOMSource source = new DOMSource(document);
      transformer.transform(source, result);
      
      fw.close();
    } catch (TransformerException te) {
      RuntimeException re = new RuntimeException();
      re.initCause(te);
      throw re;
    } catch (IOException ioe) {
      RuntimeException re = new RuntimeException();
      re.initCause(ioe);
      throw re;
    }
  }
  
  public static void main(String [] argv) throws Exception {
    new SimpleKBPDocWriter("/tmp/simplekbp").read("/u/nlp/data/TAC-KBP2010/TAC_2009_KBP_Evaluation_Reference_Knowledge_Base/data");
  }
}