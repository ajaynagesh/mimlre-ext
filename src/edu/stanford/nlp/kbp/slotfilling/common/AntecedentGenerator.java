/**
 * Sets AntecedentAnnotation for all tokens in the given corpus
 * AntecedentAnnotation is set based on the coreference annotation, but giving precedence to the KBP entity of interest, if it appears in the chain
 * @author John Bauer - main functionality
 * @author Mihai - converted this into a standalone class (extracted relevant functionality from IndexExtractor)
 */
package edu.stanford.nlp.kbp.slotfilling.common;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntTuple;
import edu.stanford.nlp.util.Pair;

public class AntecedentGenerator {
  private String entityName;
  
  private StringFinder entityFinder;
  
  private int maxSentenceLength;
  
  public AntecedentGenerator(String n, int ml) {
    this.entityName = n;
    this.entityFinder = new StringFinder(entityName);
    this.maxSentenceLength = ml;
  }
  
  /**
   * Finds and sets antecedents for all tokens in this corpus
   * @param corpus
   * @return Indices of the sentences of length < maxSentenceLength that contain the entity of interest, or a mention pointing to it
   */
  public Set<Integer> findAntecedents(Annotation corpus) {
    List<CoreMap> sentences = corpus.get(SentencesAnnotation.class);
    if (Log.levelFinerThan(Level.FINEST)) {
      for (CoreMap sentence : sentences) {
        Log.finest(sentence.get(TreeAnnotation.class).toString());
        for (CoreLabel word : sentence.get(TokensAnnotation.class)) {
          Log.finest(word.get(NamedEntityTagAnnotation.class) + " ");
        }
        Log.finest("--------------------------------");
      }

      Log.finest(corpus.get(TextAnnotation.class));
    }

    Set<Integer> goodSentences = new HashSet<Integer>();
    findContainingSentences(sentences, entityFinder, null, goodSentences);

    List<Pair<IntTuple, IntTuple>> corefGraph = getCorefGraph(corpus);
    assert(corefGraph != null);

    if (Log.levelFinerThan(Level.FINEST)) {
      Log.finest(corefGraph.toString());
      for (Pair<IntTuple, IntTuple> link : corefGraph) {
        // ms, 09202010: all coref offsets start at 1 now! => subtract 1 to get real Java indices!
        IntTuple first = link.first();
        IntTuple second = link.second();
        List<CoreLabel> firstSentence = sentences.get(first.get(0) - 1).get(TokensAnnotation.class);
        List<CoreLabel> secondSentence = sentences.get(second.get(0) - 1).get(TokensAnnotation.class);
        Log.finest(firstSentence.get(first.get(1) - 1).get(TextAnnotation.class) + " " +
                   secondSentence.get(second.get(1) - 1).get(TextAnnotation.class));
      }
    }

    Map<IntTuple, Integer> corefTupleToCluster = mapTuplesToClusters(corefGraph);
    List<List<IntTuple>> corefClusters = buildClusters(corefTupleToCluster);

    if (Log.levelFinerThan(Level.FINEST)) {
      int numClusters = corefClusters.size();
      Log.finest(numClusters + " coref clusters");
      Log.finest(corefTupleToCluster.toString());
      for (List<IntTuple> cluster : corefClusters) {
        Log.finest(cluster.toString());
      }
    }
    
    for (List<IntTuple> cluster : corefClusters) {
      Log.finest("Doing cluster " + cluster);
      String antecedent = findBestNER(sentences, cluster, entityName);

      if (antecedent == null)
        antecedent = findBestPOS(sentences, cluster, "nnp.*");
      if (antecedent == null)
        antecedent = findBestPOS(sentences, cluster, "nn.*");
      if (antecedent == null)
        antecedent = findBestPOS(sentences, cluster, "n.*");
      if (antecedent == null)
        antecedent = findBestPOS(sentences, cluster, ".*");

      if (antecedent == null)
        throw new RuntimeException("Got a document that had no part of speech tags");

      Log.fine("Found antecedent: \"" + antecedent + "\"");
      setAntecedent(sentences, cluster, antecedent);

      // If the antecedent we found matches the entity we were
      // searching for, then we want to return all of the sentences
      // that matched that particular coreference
      if (antecedent.equalsIgnoreCase(entityName)) {
        for (IntTuple tuple : cluster) {
          // ms, 09202010: all coref offsets start at 1 now! => subtract 1 to get real Java indices!
          goodSentences.add(tuple.get(0) - 1);
          Log.finest("Sentence " + (tuple.get(0) - 1) + " is useful");
        }
      }
    }
    
    return goodSentences;
  }
  
  private String findBestPOS(List<CoreMap> sentences, List<IntTuple> cluster, String posRE) {
    int firstSentence = -1;
    int firstPosition = -1;
    String antecedent = null;
    for (IntTuple tuple : cluster) {
      // ms, 09202010: all coref offsets start at 1 now! => subtract 1 to get real Java indices!
      CoreMap sentence = sentences.get(tuple.get(0) - 1);
      List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
      CoreLabel word = tokens.get(tuple.get(1) - 1);
      String tag = word.get(PartOfSpeechAnnotation.class);
      if (tag.toLowerCase().matches(posRE)) {
        Log.finest(tag + " matches " + posRE + ": " + tuple);
        if (firstSentence == -1 || firstSentence > tuple.get(0) - 1 ||
            (firstSentence == tuple.get(0) - 1 && firstPosition > tuple.get(1) - 1)) {
          firstSentence = tuple.get(0) - 1;
          firstPosition = tuple.get(1) - 1;
          antecedent = word.get(TextAnnotation.class);
        }
      }
    }
    Log.finest(firstSentence + " " + firstPosition + " " + antecedent);
    return antecedent;
  }
  
  static void setAntecedent(List<CoreMap> sentences, List<IntTuple> cluster, String antecedent) {
    for (IntTuple tuple : cluster) {
      // ms, 20100920: all coref offsets start at 1 now! => subtract 1
      // to get real Java indices!
      CoreMap sentence = sentences.get(tuple.get(0) - 1);
      sentence.get(TokensAnnotation.class).get(tuple.get(1) - 1).set(CoreAnnotations.AntecedentAnnotation.class, antecedent);
    }
  }
  
  String findBestNER(List<CoreMap> sentences, List<IntTuple> cluster, String entityName) {
    int bestNER = -1, bestStart = -1, bestEnd = -1;
    for (IntTuple tuple : cluster) {
      // ms, 20100920: all coref offsets start at 1 now! => subtract 1
      // to get real Java indices!
      CoreMap sentence = sentences.get(tuple.get(0) - 1);
      List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
      CoreLabel word = tokens.get(tuple.get(1) - 1);
      String ner = word.get(NamedEntityTagAnnotation.class);
      if (ner.equals(Constants.NER_BLANK_STRING))
        continue;

      Log.finest("Found NER: " + word + " " + ner);

      // Scroll backwards to find the first token with the same NER type
      int start = tuple.get(1) - 1;
      while (start - 1 >= 0 && 
          tokens.get(start - 1).get(NamedEntityTagAnnotation.class).equals(ner))
        --start;
      // Scroll forwards to find the last token with the same NER type
      int end = tuple.get(1) - 1;
      while (end + 1 < tokens.size() &&
          tokens.get(end + 1).get(NamedEntityTagAnnotation.class).equals(ner))
        ++end;
      end = end + 1;

      // If the named entity we found contains the entity name we were
      // given, then we consider this a hit on the entity name and return
      String potentialAntecedent = 
        " " + buildAntecedent(sentences.get(tuple.get(0) - 1), start, end) + " ";
      Log.finest(potentialAntecedent);
      if (potentialAntecedent.matches(".* " + entityName + " .*")) {
        return entityName;
      }

      // Take the newly found NER if we didn't already know of any
      // NER, or if the new one we found is longer than the previous
      // one, or if they are the same length and the new one occured
      // earlier in the document
      if (bestNER == -1 || end - start > bestStart - bestEnd ||
          (end - start == bestStart - bestEnd && 
              (tuple.get(0) - 1< bestNER ||
                  (tuple.get(0) - 1 == bestNER && start < bestStart)))) {
        bestNER = tuple.get(0) - 1;
        bestStart = start;
        bestEnd = end;
      }
    }
    
    if (bestNER == -1) return null;
    return buildAntecedent(sentences.get(bestNER), bestStart, bestEnd);
  }
  
  /**
   * Given a sentence and the start and end of the sentence, join the
   * words and return as a string
   */
  static String buildAntecedent(CoreMap sentence, int start, int end) {
    StringBuilder builder = new StringBuilder();
    for (int i = start; i < end; ++i) {
      List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
      if (i != start)
        builder.append(" ");
      builder.append(tokens.get(i).get(TextAnnotation.class));
    }
    return builder.toString();
  }

  private Set<Integer> findContainingSentences(List<CoreMap> sentences,
      StringFinder entityFinder,
      StringFinder slotKeywordFinder,
      Set<Integer> goodSentences) {
    // Add any sentence that contains a sequence of tokens matching
    // the entity we care about.  Note that this block works with a
    // minimal pipeline (just tokenization and sentence splitting)!
    for (int sentenceIndex = 0; sentenceIndex < sentences.size(); ++sentenceIndex) {
      CoreMap sentence = sentences.get(sentenceIndex);
      assert(sentence != null);
      List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
      assert(tokens != null);

      if ((maxSentenceLength <= 0 || tokens.size() <= maxSentenceLength) && 
          entityFinder.matches(sentence) &&
          (slotKeywordFinder == null || slotKeywordFinder.matches(sentence)))
        goodSentences.add(sentenceIndex);
    }

    return goodSentences;
  }
  
  static private List<Pair<IntTuple, IntTuple>> getCorefGraph(Annotation annotation) {
    List<Pair<IntTuple, IntTuple>> graph = 
      annotation.get(CorefCoreAnnotations.CorefGraphAnnotation.class);
    if (graph == null) 
      return Collections.<Pair<IntTuple, IntTuple>>emptyList();
    else
      return graph;
  }
  
  // TODO: it doesn't make sense to have this separate from
  // buildClustersMap, since we're doing the same work in this method
  public static Map<IntTuple, Integer> mapTuplesToClusters(List<Pair<IntTuple, IntTuple>> corefGraph) {
    Map<IntTuple, Integer> corefTupleToCluster = 
      new HashMap<IntTuple, Integer>();
    List<List<IntTuple>> clusterToTuples = new ArrayList<List<IntTuple>>();
    

    int numClusters = 0;
    for (Pair<IntTuple, IntTuple> link : corefGraph) {
      IntTuple first = link.first();
      IntTuple second = link.second();
      if (corefTupleToCluster.containsKey(first) &&
          corefTupleToCluster.containsKey(second)) {
        if (!corefTupleToCluster.get(first).equals(corefTupleToCluster.get(second))) {
          int moveFrom, moveTo;
          if (corefTupleToCluster.get(first) > 
              corefTupleToCluster.get(second)) {
            moveFrom = corefTupleToCluster.get(first);
            moveTo = corefTupleToCluster.get(second);
          } else {
            moveTo = corefTupleToCluster.get(first);
            moveFrom = corefTupleToCluster.get(second);
          }
          for (IntTuple tuple : clusterToTuples.get(moveFrom)) {
            corefTupleToCluster.put(tuple, moveTo);
          }
          clusterToTuples.get(moveTo).addAll(clusterToTuples.get(moveFrom));
          clusterToTuples.get(moveFrom).clear();
          // now, we have created a gap...  fill it in
          --numClusters;
          if (moveFrom < numClusters) {
            for (IntTuple tuple : clusterToTuples.get(numClusters)) {
              corefTupleToCluster.put(tuple, moveFrom);
            }
            clusterToTuples.get(moveFrom).addAll(clusterToTuples.get(numClusters));
          }
          clusterToTuples.remove(numClusters);
        }
      } else if (corefTupleToCluster.containsKey(first)) {
        corefTupleToCluster.put(second, corefTupleToCluster.get(first));
        clusterToTuples.get(corefTupleToCluster.get(first)).add(second);
      } else if (corefTupleToCluster.containsKey(second)) {          
        corefTupleToCluster.put(first, corefTupleToCluster.get(second));
        clusterToTuples.get(corefTupleToCluster.get(second)).add(first);
      } else {
        // most annoying bug ever: caused by finding that Integers
        // come from an object pool if they are small enough, eg <
        // 128, but are created as separate objects if they are >=
        // 128, in which case x == y no longer works.
        corefTupleToCluster.put(first, numClusters);
        corefTupleToCluster.put(second, numClusters);
        clusterToTuples.add(new ArrayList<IntTuple>());
        clusterToTuples.get(numClusters).add(first);
        clusterToTuples.get(numClusters).add(second);
        ++numClusters;
      }
    }

    return corefTupleToCluster;
  }
  
  public static List<List<IntTuple>> buildClusters(Map<IntTuple, Integer> corefTupleToCluster) {
    List<List<IntTuple>> corefClusters = new ArrayList<List<IntTuple>>();

    for (Map.Entry<IntTuple, Integer> entry : corefTupleToCluster.entrySet()) {
      for (int i = corefClusters.size(); i <= entry.getValue(); ++i) {
        corefClusters.add(new ArrayList<IntTuple>());
      }
      corefClusters.get(entry.getValue()).add(entry.getKey());
    }

    return corefClusters;
  }
}
