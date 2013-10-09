/**
 * Stores the information required to identify an <entity, slot> tuples and all its mentions in text as datums
 * This is the input to the classifier in KBPEvaluator
 * See DatumAndMention.toKBPTuple for the conversion from DatumAndMention to KBPTuple
 */
package edu.stanford.nlp.kbp.slotfilling.common;

import java.util.*;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.SourceIndexAnnotation;
import edu.stanford.nlp.kbp.slotfilling.index.DocidFinder;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.CoreAnnotations.DocIDAnnotation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.Pair;

public class KBPTuple {
  /** Used only when KBPEvaluator.diagnosticMode == true */
  private final List<String> goldMentionLabels;
  
  /** Each datum corresponds to a distinct mention of this tuple in text */
  private final List<Datum<String, String>> datums;
  
  private final String entityId;
  private final String entityName;
  private final String entityType;
  private final String slotType;
  private final String slotValue;
  private final Set<String> normalizedSlotValues;
  private final String docid;
  private final String indexName;
  
  public int size() { return datums.size(); }
  public String goldMentionLabel(int i) { return goldMentionLabels.get(i); }
  public Datum<String, String> datum(int i) { return datums.get(i); }
  public String entityId() { return entityId; }
  public String entityName() { return entityName; }
  public String entityType() { return entityType; }
  public EntityType entityKBPType() {
    EntityType type = EntityType.PERSON;
    if (entityType.contains("ORG")) type = EntityType.ORGANIZATION;
    return type;
  }
  public String slotType() { return slotType; }
  public String slotValue() { return slotValue; }
  public Set<String> normalizedSlotValues() { return normalizedSlotValues; }
  public String docid() { return docid; }
  public String indexName() { return indexName; }

  public KBPTuple(
          String entityId,
          String entityName,
          String entityType,
          String slotType,
          String slotValue,
          Set<String> normalizedSlotValues,
          String indexName,
          String docid,
          List<String> goldMentionLabels,
          List<Datum<String, String>> datums) {
    this.entityId = entityId;
    this.entityName = entityName;
    this.entityType = entityType;
    this.slotType = slotType;
    this.slotValue = slotValue;
    this.normalizedSlotValues = normalizedSlotValues;
    this.indexName = indexName;
    this.docid = docid;
    this.goldMentionLabels = goldMentionLabels;
    this.datums = datums;
  }
  
  public KBPTuple(List<DatumAndMention> mentions,
                  Map<String, KBPEntity> originalTask,
                  DocidFinder docidFinder) {
    DatumAndMention firstRelationMention = mentions.iterator().next();
    EntityMention firstEntityMention = (EntityMention) firstRelationMention.mention().getArg(0);
    entityId = getKbpId(firstEntityMention);
    entityType = firstEntityMention.getType();
    // bug: firstEntityMention may contain an NP that is coreferent w/ the entity but has a different extent!
    // entityName = firstEntityMention.getExtentString().replaceAll("\t+", " ");
    entityName = originalTask.get(entityId).name;
    slotType = chooseMajoritySlotNE(mentions);
    
    // pick the longest extent as the slot value
    String bestSlotValue = "";
    // String bestSlotValueNE = "";
    normalizedSlotValues = new HashSet<String>();
    for (DatumAndMention m : mentions) {
      String ext = m.mention().getArg(1).getExtentString();
      if (ext.length() > bestSlotValue.length()) {
        bestSlotValue = ext;
        // bestSlotValueNE = m.mention().getArg(1).getType();
      }
      normalizedSlotValues().add(m.mention().getNormalizedSlotValue());
    }
    slotValue = bestSlotValue.replaceAll("\t+", " ");
    // slotValueNE = bestSlotValueNE;

    goldMentionLabels = new ArrayList<String>();
    datums = new ArrayList<Datum<String,String>>();
    for(DatumAndMention mention: mentions) {
      goldMentionLabels.add(mention.mention().getType());
      datums.add(mention.datum());
    }
    
    //
    // traverse all relation mentions for this slot to search for an index
    // if the official index is present, pick that; then pick the non-web one;
    // then pick web
    //
    String bestDocid = null;
    String bestIndexName = null;
    for (DatumAndMention mention : mentions) {
      String docid = mention.mention().getArg(0).getSentence().get(DocIDAnnotation.class);
      String indexName = Constants.getIndexPath(mention.mention().getArg(0).getSentence().get(SourceIndexAnnotation.class));
      assert (indexName != null);
      if (indexName.equals(docidFinder.getSource())) {
        bestDocid = docid;
        bestIndexName = indexName;
        break;
      } else if (!indexName.equals(Constants.WEBINDEX_NAME)
          && (bestIndexName == null || bestIndexName.equals(Constants.WEBINDEX_NAME))) {
        bestDocid = docid;
        bestIndexName = indexName;
      } else if (indexName.equals(Constants.WEBINDEX_NAME) && bestIndexName == null) {
        bestIndexName = indexName;
        bestDocid = docid;
      }
    }

    // if we did not find a docid in the official index, try again by directly searching the index
    String officialDocid;
    if(! bestIndexName.equals(docidFinder.getSource()) &&
      (officialDocid = docidFinder.findBestDocid(entityName, slotValue)) != null) {
      // found one!
      Log.severe("Found an official docid in the index for query: [" + entityName + "] + [" + slotValue + "]");
      docid = officialDocid;
      indexName = docidFinder.getSource();
    } else {
      // not found, use whatever, it doesn't really matter
      docid = bestDocid;
      indexName = bestIndexName;
    }
  }
  
  private static String chooseMajoritySlotNE(Collection<DatumAndMention> mentions) {
    Counter<String> neCounts = new ClassicCounter<String>();
    for (DatumAndMention m : mentions)
      neCounts.incrementCount(m.mention().getArg(1).getType());
    List<Pair<String, Double>> sortedCounts = Counters.toDescendingMagnitudeSortedListWithCounts(neCounts);
    String firstType = mentions.iterator().next().mention().getArg(1).getType();
    double firstCount = neCounts.getCount(firstType);
    if (firstCount == sortedCounts.get(0).second())
      return firstType;
    else
      return sortedCounts.get(0).first();
  }

  public static String getKbpId(EntityMention em) {
    int kbpPos = em.getObjectId().indexOf("KBP");
    assert (kbpPos > 0);
    String id = em.getObjectId().substring(kbpPos + 3);
    return id;
  }
}
