package edu.stanford.nlp.kbp.slotfilling.index;

import java.util.Stack;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

/**
 * A class which is a DefaultHandler for SAX XML parsers, but also
 * keeps track of a stack of tags as it goes.
 *
 * @author John Bauer
 */
public class TagStackXmlHandler extends DefaultHandler {
  protected Stack<String> tagStack = new Stack<String>();
  
  @Override
  public void startElement(String uri, String localName, 
                           String qName, Attributes attributes)
    throws SAXException
  {
    String name = ((!localName.equals("")) ? localName : qName);
    tagStack.push(name);
  }
  
  @Override
  public void endElement(String uri, String localName, String qName) 
    throws SAXException
  {
    tagStack.pop();
  }
  
  public final boolean matchesTags(String[] tags) {
    if (tagStack.size() != tags.length)
      return false;
    
    for (int i = 0; i < tagStack.size(); ++i) {
      if (!tagStack.get(i).equalsIgnoreCase(tags[i]))
        return false;
    }
    
    return true;
  } 
}