package edu.stanford.nlp.kbp.slotfilling.multir;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.GZIPInputStream;

import edu.stanford.nlp.kbp.slotfilling.classify.MultiLabelDataset;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.multir.DocumentProtos.Relation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;

/**
 * Converts Hoffmann's data in protobuf format to our MultiLabelDataset
 * @author Mihai
 *
 */
public class ProtobufToMultiLabelDataset {
  static class RelationAndMentions {
    String arg1;
    String arg2;
    Set<String> posLabels;
    Set<String> negLabels;
    List<Mention> mentions;
    
    public RelationAndMentions(String types, String a1, String a2) {
      arg1 = a1;
      arg2 = a2;
      String [] rels = types.split(",");
      posLabels = new HashSet<String>();
      for(String r: rels){
        if(! r.equals("NA")) posLabels.add(r.trim());
      }
      negLabels = new HashSet<String>(); // will be populated later
      mentions = new ArrayList<Mention>();
    }
  };
  
  static class Mention {
    List<String> features;
    public Mention(List<String> feats) {
      features = feats;
    }
  }
  
	public static void main(String[] args) throws Exception {
	  String input = args[0];
	  
	  InputStream is = new GZIPInputStream(
        new BufferedInputStream
        (new FileInputStream(input)));
	  
	  toMultiLabelDataset(is);
	  is.close();
	}
	
	public static MultiLabelDataset<String, String> toMultiLabelDataset(InputStream is) throws IOException {
	  List<RelationAndMentions> relations = toRelations(is, true);
	  MultiLabelDataset<String, String> dataset = toDataset(relations);
	  return dataset;
	}
	
	public static void toDatums(InputStream is,
	    List<List<Collection<String>>> relationFeatures,
	    List<Set<String>> labels) throws IOException {
	  List<RelationAndMentions> relations = toRelations(is, false);
	  toDatums(relations, relationFeatures, labels);
	}
	
	private static void toDatums(List<RelationAndMentions> relations,
	    List<List<Collection<String>>> relationFeatures,
      List<Set<String>> labels) {
    for(RelationAndMentions rel: relations) {
      labels.add(rel.posLabels);
      List<Collection<String>> mentionFeatures = new ArrayList<Collection<String>>();
      for(int i = 0; i < rel.mentions.size(); i ++){
        mentionFeatures.add(rel.mentions.get(i).features);
      }
      relationFeatures.add(mentionFeatures);
    }
    assert(labels.size() == relationFeatures.size());
	}
	
	public static List<RelationAndMentions> toRelations(InputStream is, boolean generateNegativeLabels) throws IOException {
	  //
	  // Parse the protobuf
	  //
    // all relations are stored here
    List<RelationAndMentions> relations = new ArrayList<RelationAndMentions>();
    // all known relations (without NIL)
    Set<String> relTypes = new HashSet<String>();
    Map<String, Map<String, Set<String>>> knownRelationsPerEntity = 
      new HashMap<String, Map<String,Set<String>>>();
    Counter<Integer> labelCountHisto = new ClassicCounter<Integer>();
    Relation r = null;
    while ((r = Relation.parseDelimitedFrom(is)) != null) {
      RelationAndMentions relation = new RelationAndMentions(
          r.getRelType(), r.getSourceGuid(), r.getDestGuid());
      labelCountHisto.incrementCount(relation.posLabels.size());
      relTypes.addAll(relation.posLabels);
      relations.add(relation);

      for(int i = 0; i < r.getMentionCount(); i ++) {
        DocumentProtos.Relation.RelationMentionRef mention = r.getMention(i);
        // String s = mention.getSentence();
        relation.mentions.add(new Mention(mention.getFeatureList()));
      }
      
      for(String l: relation.posLabels) {
        addKnownRelation(relation.arg1, relation.arg2, l, knownRelationsPerEntity);
      }
    }
    Log.severe("Loaded " + relations.size() + " relations.");
    Log.severe("Found " + relTypes.size() + " relation types: " + relTypes);
    Log.severe("Label count histogram: " + labelCountHisto);
    
    Counter<Integer> slotCountHisto = new ClassicCounter<Integer>();
    for(String e: knownRelationsPerEntity.keySet()) {
      slotCountHisto.incrementCount(knownRelationsPerEntity.get(e).size());
    }
    Log.severe("Slot count histogram: " + slotCountHisto);
    int negativesWithKnownPositivesCount = 0, totalNegatives = 0;
    for(RelationAndMentions rel: relations) {
      if(rel.posLabels.size() == 0) {
        if(knownRelationsPerEntity.get(rel.arg1) != null &&
           knownRelationsPerEntity.get(rel.arg1).size() > 0) {
          negativesWithKnownPositivesCount ++;
        }
        totalNegatives ++;
      }
    }
    Log.severe("Found " + negativesWithKnownPositivesCount + "/" + totalNegatives +
        " negative examples with at least one known relation for arg1.");
    
    Counter<Integer> mentionCountHisto = new ClassicCounter<Integer>();
    for(RelationAndMentions rel: relations) {
      mentionCountHisto.incrementCount(rel.mentions.size());
      if(rel.mentions.size() > 100) 
        Log.fine("Large relation: " + rel.mentions.size() + "\t" + rel.posLabels);
    }
    Log.severe("Mention count histogram: " + mentionCountHisto);
    
    //
    // Detect the known negatives for each source entity
    //
    if(generateNegativeLabels) {
      for(RelationAndMentions rel: relations) {
        Set<String> negatives = new HashSet<String>(relTypes);
        negatives.removeAll(rel.posLabels);
        rel.negLabels = negatives;
      }
    }
    
    return relations;
	}
   
	private static MultiLabelDataset<String, String> toDataset(List<RelationAndMentions> relations) {
    int [][][] data = new int[relations.size()][][];
    Index<String> featureIndex = new HashIndex<String>();
    Index<String> labelIndex = new HashIndex<String>();
    Set<Integer> [] posLabels = ErasureUtils.<Set<Integer> []>uncheckedCast(new Set[relations.size()]);
    Set<Integer> [] negLabels = ErasureUtils.<Set<Integer> []>uncheckedCast(new Set[relations.size()]);
    
    int offset = 0, posCount = 0;
    for(RelationAndMentions rel: relations) {
      Set<Integer> pos = new HashSet<Integer>();
      Set<Integer> neg = new HashSet<Integer>();
      for(String l: rel.posLabels) {
        pos.add(labelIndex.indexOf(l, true));
      }
      for(String l: rel.negLabels) {
        neg.add(labelIndex.indexOf(l, true));
      }
      posLabels[offset] = pos;
      negLabels[offset] = neg;
      int [][] group = new int[rel.mentions.size()][];
      for(int i = 0; i < rel.mentions.size(); i ++){
        List<String> sfeats = rel.mentions.get(i).features;
        int [] features = new int[sfeats.size()];
        for(int j = 0; j < sfeats.size(); j ++) {
          features[j] = featureIndex.indexOf(sfeats.get(j), true); 
        }
        group[i] = features;
      }
      data[offset] = group;
      posCount += posLabels[offset].size();
      offset ++;
    }
    
    Log.severe("Creating a dataset with " + data.length + " datums, out of which " + posCount + " are positive.");
    MultiLabelDataset<String, String> dataset = new MultiLabelDataset<String, String>(
        data, featureIndex, labelIndex, posLabels, negLabels);
    return dataset;
	}
	
	private static void addKnownRelation(String arg1, String arg2, String label,
	    Map<String, Map<String, Set<String>>> knownRelationsPerEntity) {
	  Map<String, Set<String>> myRels = knownRelationsPerEntity.get(arg1);
	  if(myRels == null) {
	    myRels = new HashMap<String, Set<String>>();
	    knownRelationsPerEntity.put(arg1, myRels);
	  }
	  Set<String> mySlots = myRels.get(label);
	  if(mySlots == null) {
	    mySlots = new HashSet<String>();
	    myRels.put(label, mySlots);
	  }
	  mySlots.add(arg2);
	}
}
