package edu.stanford.nlp.kbp.slotfilling.index;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasIndex;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.Label;
import edu.stanford.nlp.pipeline.ParserAnnotatorUtils;
import edu.stanford.nlp.trees.LabeledScoredTreeNode;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphFactory;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.*;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.CoreMap;

import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.GrammaticalRelation.Language;
import edu.stanford.nlp.trees.GrammaticalRelation.
         GrammaticalRelationAnnotation;

@SuppressWarnings("unused")
public class CoreMapCombiner {
  public static final String FAKE_ROOT_NAME = "fakeroot";
  public static final GrammaticalRelation FAKE_ROOT =
    new GrammaticalRelation(Language.Any, FAKE_ROOT_NAME, FAKE_ROOT_NAME, 
                            FakeRootGRAnnotation.class, null);

  public static class FakeRootGRAnnotation
    extends GrammaticalRelationAnnotation{ }

  public static CoreMap combine(List<CoreMap> sentences,
                                int beginIndex, int endIndex, int rootIndex) {
    if (endIndex == beginIndex + 1)
      return sentences.get(beginIndex);

    if (rootIndex >= endIndex || rootIndex < beginIndex)
      throw new IllegalArgumentException("The rootIndex " + rootIndex +
                                         " is outside the range " +
                                         beginIndex + " ... " + endIndex);

    CoreMap newSentence = new ArrayCoreMap();

    //
    // merge tokens
    //
    List<CoreLabel> newTokens = new ArrayList<CoreLabel>();
    for (int i = beginIndex; i < endIndex; ++i) {
      CoreMap sentence = sentences.get(i);
      newTokens.addAll(sentence.get(TokensAnnotation.class));
    }
    newSentence.set(TokensAnnotation.class, newTokens);

    //
    // merge constituent trees
    // note: in some weird cases (e.g., serialization error during caching), some sentences do not have trees
    //
    boolean foundAllTrees = true;
    for (int i = beginIndex; i < endIndex; ++i) {
      CoreMap sentence = sentences.get(i);
      Tree tree = sentence.get(TreeAnnotation.class);
      if(tree == null) {
        foundAllTrees = false;
        break;
      }
    }
    if(foundAllTrees){
      Tree newTree = new LabeledScoredTreeNode();
      List<Tree> children = new ArrayList<Tree>();
      for (int i = beginIndex; i < endIndex; ++i) {
        CoreMap sentence = sentences.get(i);
        Tree tree = sentence.get(TreeAnnotation.class);
        // This next command would barf if we had a class that does not
        // allow for setting of children.  Hopefully not the case.
        for (Tree child : tree.children()) {
          children.add(child.deepCopy());
        }
      }
      newTree.setChildren(children);
      CoreLabel fakeRootLabel = new CoreLabel();
      fakeRootLabel.setValue("ROOT");
      newTree.setLabel(fakeRootLabel);
      newSentence.set(TreeAnnotation.class, newTree);

      // The leaves of the tree are incorrectly indexed at this point
      List<Label> leaves = newTree.yield();
      if (leaves.size() != newTokens.size()) {
        // TODO: we should figure out why this happens.  It's either a
        // bug in the tokenizer or in the parser
        return null;
      }
      int index = 1;
      for (Label leaf : leaves) {
        if (!(leaf instanceof HasIndex))
          throw new IllegalArgumentException("Leaves should be HasIndexs");
        ((HasIndex) leaf).setIndex(index);
        ++index;
      }
    }

    //
    // merge text
    //
    StringBuilder newText = new StringBuilder();
    for (int i = beginIndex; i < endIndex; ++i) {
      CoreMap sentence = sentences.get(i);
      if (i > beginIndex)
        newText.append(" ");
      newText.append(sentence.get(TextAnnotation.class));
    }
    newSentence.set(TextAnnotation.class, newText.toString());

    //
    // merge dependency graphs
    // note: in some weird cases (e.g., serialization error during caching), some sentences do not have dep graphs
    //
    boolean foundAllGraphs = true;
    for (int i = beginIndex; i < endIndex; ++i) {
      CoreMap sentence = sentences.get(i);
      SemanticGraph originalGraph = 
        sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
      if(originalGraph == null){
        foundAllGraphs = false;
        break;
      }
    }
    if(foundAllGraphs){
      // Gather lists of all the SemanticGraphs and the token lengths for each sentence.
      List<SemanticGraph> graphs = new ArrayList<SemanticGraph>();
      List<Integer> lengths = new ArrayList<Integer>();
      int rootOffset = 0;
      int graphRoot = -1;
      for (int i = beginIndex; i < endIndex; ++i) {
        CoreMap sentence = sentences.get(i);
        SemanticGraph originalGraph = 
          sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
        graphs.add(originalGraph);
        int length = sentence.get(TokensAnnotation.class).size(); 
        lengths.add(length);
        if (i < rootIndex) {
          rootOffset += length;
        } else if (i == rootIndex) {
          // Also, keep track of the index of the root node of the
          // "root" sentence.  If such a node does not exist, we can't
          // construct a graph, so we skip this potential megasentence.
          if (originalGraph.getRoots() == null || 
              originalGraph.getRoots().size() == 0) {
            return null;
          }
          graphRoot = originalGraph.getFirstRoot().index() + rootOffset;
        }

        //System.out.println("----------------------------------");
        //System.out.println(originalGraph);
      }

      // This should be impossible, since we know rootIndex sentence has
      // at least one sentence
      if (graphRoot == -1) {
        throw new AssertionError("Failed to find the root of the graph");
      }

      // Having gotten these lists, construct a megagraph using deep
      // copies of the structure
      SemanticGraph newGraph = 
        SemanticGraphFactory.deepCopyFromGraphs(graphs, lengths);

      //System.out.println("**********************************");
      //System.out.println(newGraph);
      //System.out.println("**********************************");

      // now, we set up links from the root node to everything else that
      // was called a root in the original graphs
      // first we get the IndexedWord representing the root...
      IndexedWord graphRootWord = null;
      for (IndexedWord root : newGraph.getRoots()) {
        if (root.index() == graphRoot) {
          graphRootWord = root;
          break;
        }
      }
      // (which should never fail)
      if (graphRootWord == null) {
        throw new AssertionError("Failed to find the root of the graph");
      }
      // and then add fakeroot dependencies to all the other roots
      for (IndexedWord root : newGraph.getRoots()) {
        if (root.index() != graphRoot) {
          newGraph.addEdge(graphRootWord, root, FAKE_ROOT, 1.0);
        }
      }
      newGraph.setRoot(graphRootWord);

      // yay, new graph
      newSentence.set(CollapsedCCProcessedDependenciesAnnotation.class, 
          newGraph);

      //System.out.println(newGraph);
    }

    return newSentence;
  }

  private CoreMapCombiner() {} // static methods only
}
