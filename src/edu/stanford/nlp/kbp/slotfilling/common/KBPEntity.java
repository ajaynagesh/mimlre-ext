package edu.stanford.nlp.kbp.slotfilling.common;

import java.io.Serializable;
import java.util.Comparator;
import java.util.Set;

import edu.stanford.nlp.util.ArrayUtils;


public class KBPEntity implements Serializable{

  private static final long serialVersionUID = 1L;
  public EntityType type;
  public String name;
  public String id;
  public String docid;
  public String queryId;
  public Set<String> ignoredSlots;
  
  @Override
  public int hashCode() {
    assert(id != null);
    return id.hashCode();
  }
  
  @Override
  public boolean equals(Object obj) {
    if(obj instanceof KBPEntity){
      KBPEntity em = (KBPEntity) obj;
      assert(id != null && em.id != null);
      if(em.id.equals(id)) return true;
    }
    return false;
  }

  @Override
  public String toString() {
    String s = type + ":" + name;
    if(id != null) s += " (" + id + "," + queryId + ")";
    return s;
  }  

  public static class QueryIdSorter implements Comparator<KBPEntity> {
    public int compare(KBPEntity first, KBPEntity second) {
      // queryId should be unique, so there shouldn't be a need for a
      // fallback comparison
      return first.queryId.compareTo(second.queryId);
    }
  }
  
  /**
   * Sort first alphabetically by entity name, then by query ID, then by entity ID, then by type.
   */
  public static class AlphabeticSorter implements Comparator<KBPEntity> {
    public int compare(KBPEntity first, KBPEntity second) {
      return ArrayUtils.compareArrays(extractFields(first), extractFields(second));
    }

    private String[] extractFields(KBPEntity entity) {
      String queryId = entity.queryId;
      if (queryId == null) {
        queryId = "(null)";
      }
      String id = entity.id;
      if (id == null) {
        id = "(null)";
      }
      return new String[] { entity.name, queryId, id, entity.type.toString() };
    }
  }
}
