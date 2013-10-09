package edu.stanford.nlp.kbp.slotfilling.index;

import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

import org.xml.sax.Attributes;
import org.xml.sax.helpers.DefaultHandler;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.util.StringUtils;

/**
 * This class defines a program that iterates over a set of xml files
 * and coounts how many different kinds of tag nesting appear.
 * Repeated instances of the same nesting in one file only count once,
 * whereas two different files that contain the same nesting count as
 * two instances.
 * <br>
 * This class is not thread safe, but it would be easy enough to make
 * it so if that were somehow useful.
 * <br>
 * To run, simply add a list of files and/or directories at the
 * command line.  The program will recursively go through each
 * directory, looking for files to process.  
 * <br>
 * It doesn't do anything to distinguish xml from non-xml files, but
 * plain text files, for one, would be a no-op.
 */
public class CountNestedTags extends KBPFileProcessor {
  /**
   * This would need to be a thread-safe counter if you want to make
   * this program thread-safe
   */
  public final ClassicCounter<String> nestedTagCounts = new ClassicCounter<String>();

  /**
   * If you want this class to be thread safe for some reason, you
   * would need to create a new NestedTagHandler for each file rather
   * than reusing the same one over and over
   */
  final NestedTagHandler handler = new NestedTagHandler();

  protected DefaultHandler getHandler() {
    return handler;
  }

  /**
   * To run: java &lt;program name&gt; [filename...]
   */
  static public void main(String[] args) {
    CountNestedTags tags = new CountNestedTags();
    tags.recursiveProcess(args);
    System.out.println(tags.nestedTagCounts);
  }
  
  /**
   * This handler keeps track of unique tag nesting structures seen in
   * a hash set, and then adds that hash set to the containing class's
   * counter at the end of a document.  
   * <br>
   * Because it is an inner class, it has access to the
   * nestedTagCounts field above.
   */
  class NestedTagHandler extends KbpXmlHandler {
    Set<String> documentNestedTags = new HashSet<String>();
    Stack<String> tagStack = new Stack<String>();

    @Override
    public void startDocument() {
      tagStack.clear();
      documentNestedTags.clear();
    }

    @Override
    public void endDocument() {
      nestedTagCounts.addAll(new ClassicCounter<String>(documentNestedTags));
    }

    @Override
    public void startElement(String uri, String localName, 
                             String qName, Attributes attributes) 
    {
      String name = ((!localName.equals("")) ? localName : qName);
      tagStack.push(name);
      documentNestedTags.add(StringUtils.join(tagStack, "-"));
    }

    @Override
    public void endElement(String uri, String localName, 
                           String qName) {
      tagStack.pop();
    }
  }
}
