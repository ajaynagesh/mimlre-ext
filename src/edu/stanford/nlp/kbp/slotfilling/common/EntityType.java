
package edu.stanford.nlp.kbp.slotfilling.common;

import java.util.HashMap;
import java.util.Map;

/**
 * Different kinds of entities you might want to search for in the KBP
 * data.  Other packages may behave differently depending on the
 * entity type you request.
 */
public enum EntityType {
  PERSON          ("PER"), 
  ORGANIZATION    ("ORG");

  private final String xmlRepresentation;
  EntityType(String xml) {
    this.xmlRepresentation = xml;
  }

  private static final Map<String, EntityType> xmlToEntity = 
    new HashMap<String, EntityType>();
  static {
    for (EntityType entity : values()) {
      xmlToEntity.put(entity.xmlRepresentation, entity);
    }
  }

  static public EntityType fromXmlRepresentation(String xml) {
    return xmlToEntity.get(xml);
  }
}