package edu.stanford.nlp.kbp.slotfilling.index;

/**
 * A class which keeps track of the tag sequences which determine the 
 * parts of a KBP document we care about.
 *
 * @author John Bauer
 */
public class KbpXmlHandler extends TagStackXmlHandler {
  static final String[] DATETIME_TAGS = {"doc", "datetime"};
  static final String[] DOCID_TAGS = {"doc", "docid"};
  static final String[] TEXT_TAGS = {"doc", "body", "text"};
  static final String[] HEADLINE_TAGS = {"doc", "body", "headline"};

  static final String[] SPEAKER_TAGS = {"doc", "body", "text", 
                                        "turn", "speaker"};
}
  
