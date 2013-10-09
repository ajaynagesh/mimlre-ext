
package edu.stanford.nlp.kbp.slotfilling.index;

import org.apache.lucene.document.Field;


/**
 * This enum contains the different fields we keep in the Lucene
 * repository.  When text names are needed, such as for field names,
 * the strings are the enum's .toString() method.
 */
public enum KBPField {
  DATETIME  ("datetime",  Field.Index.NOT_ANALYZED), 
  DOCID     ("date",      Field.Index.NOT_ANALYZED),
  HEADLINE  ("headline",  Field.Index.ANALYZED), 
  TEXT      ("text",      Field.Index.ANALYZED),
  WIKITITLE ("title",     Field.Index.NOT_ANALYZED), 
  WIKICONTENT ("content", Field.Index.ANALYZED),
  COREMAP   ("coremap",   Field.Index.NOT_ANALYZED);

  private final String fieldName;
  private final Field.Index indexingStrategy;
  KBPField(String fieldName, Field.Index indexingStrategy) {
    this.fieldName = fieldName;
    this.indexingStrategy = indexingStrategy;
  }

  public String fieldName() {
    return fieldName;
  }
  
  public Field.Index indexingStrategy() {
    return indexingStrategy;
  }
}

