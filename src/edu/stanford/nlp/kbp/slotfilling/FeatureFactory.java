package edu.stanford.nlp.kbp.slotfilling;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations.GenderAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations.TriggerAnnotation;
import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.kbp.slotfilling.common.EntityType;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Utils;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.SourceIndexAnnotation;
import edu.stanford.nlp.kbp.slotfilling.index.CoreMapCombiner;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.trees.EnglishGrammaticalRelations;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.trees.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;

public class FeatureFactory implements Serializable {
  private static final long serialVersionUID = -7376668998622546620L;

  private static final Logger logger = Logger.getLogger(FeatureFactory.class.getName());

  static {
  	logger.setLevel(Level.INFO);
  }

  public static enum DEPENDENCY_TYPE {
    BASIC, COLLAPSED, COLLAPSED_CCPROCESSED
  }

  private static final List<String> dependencyFeatures = Collections.unmodifiableList(Arrays.asList(new String [] {
      "dependency_path_lowlevel","dependency_path_length","dependency_path_length_binary",
      "verb_in_dependency_path","dependency_path","dependency_path_words","dependency_paths_to_verb",
      "dependency_path_stubs_to_verb",
      "dependency_path_POS_unigrams",
      "dependency_path_word_n_grams",
      "dependency_path_POS_n_grams",
      "dependency_path_edge_n_grams","dependency_path_edge_lowlevel_n_grams",
      "dependency_path_edge-node-edge-grams","dependency_path_edge-node-edge-grams_lowlevel",
      "dependency_path_node-edge-node-grams","dependency_path_node-edge-node-grams_lowlevel",
      "dependency_path_directed_bigrams",
      "dependency_path_edge_unigrams",
      "dependency_path_trigger",
      "different_sentences",
      }));

  /** Which dependencies to use for feature extraction */
  protected DEPENDENCY_TYPE dependencyType;

  protected List<String> featureList;

  /** If true, it does not create any lexicalized features from the first argument (needed for KBP) */
  protected boolean doNotLexicalizeFirstArg;

  public FeatureFactory(String... featureList) {
    this.doNotLexicalizeFirstArg = false;
    this.featureList = Collections.unmodifiableList(Arrays.asList(featureList));
    this.dependencyType = DEPENDENCY_TYPE.COLLAPSED_CCPROCESSED;
  }

  public void setDoNotLexicalizeFirstArgument(boolean doNotLexicalizeFirstArg) {
    this.doNotLexicalizeFirstArg = doNotLexicalizeFirstArg;
  }

  public Datum<String,String> createDatum(RelationMention rel) {
    if (rel.getArgs().size() != 2) {
      return null;
    }

    Collection<String> features = new ArrayList<String>();
    addFeatures(features, rel, featureList);
    String labelString = rel.getType();
    return new BasicDatum<String, String>(features, labelString);
  }

  public Datum<String,String> createDatum(RelationMention rel, String positiveLabel) {
    if (rel.getArgs().size() != 2) {
      return null;
    }

    Collection<String> features = new ArrayList<String>();
    addFeatures(features, rel, featureList);
    String labelString = rel.getType();
    if(! labelString.equals(positiveLabel)) labelString = RelationMention.UNRELATED;
    return new BasicDatum<String, String>(features, labelString);
  }

  // BEGIN GRAFT
  public boolean addFeatures(Collection<String> features, RelationMention rel, List<String> types) {
    Collection<String> rawFeatures = new ArrayList<String>();
    boolean retCode = addFeaturesRaw(rawFeatures, rel, types);
    postProcessFeatures(rawFeatures, features);
    return retCode;
  }

  private void postProcessFeatures(Collection<String> src, Collection<String> dst) {
    for(String rawName: src){
      String cleanName = postProcessFeature(rawName);
      dst.add(cleanName);
    }
  }

  private String postProcessFeature(String feat) {
    // do not allow spaces in a feature
    feat = feat.replaceAll("\\s+", "_");

    return feat;
  }
  // END GRAFT

  /**
   * Creates all features for the datum corresponding to this relation mention
   * Note: this assumes binary relations where both arguments are EntityMention
   * @param features Stores all features
   * @param rel The mention
   * @param types Comma separated list of feature classes to use
   */
  private boolean addFeaturesRaw(Collection<String> features, RelationMention rel, List<String> types) {
    // sanity checks: must have two arguments, and each must be an entity mention
    if(rel.getArgs().size() != 2) return false;
    if(! (rel.getArg(0) instanceof EntityMention)) return false;
    if(! (rel.getArg(1) instanceof EntityMention)) return false;

    EntityMention arg0 = (EntityMention) rel.getArg(0);
    EntityMention arg1 = (EntityMention) rel.getArg(1);
    String arg0Type = findTrueEntityType(arg0.getType(), rel.getType());

  	Tree tree = rel.getSentence().get(TreeAnnotation.class);
  	if(tree == null){
  	  throw new RuntimeException("ERROR: Relation extraction requires full syntactic analysis!");
  	}
  	List<Tree> leaves = tree.getLeaves();
  	List<CoreLabel> tokens = rel.getSentence().get(TokensAnnotation.class);

  	// Checklist keeps track of which features have been handled by an if clause
  	// Should be empty after all the clauses have been gone through.
  	List<String> checklist = new ArrayList<String>(types);

  	// arg_type: concatenation of the entity types of the args, e.g.
  	// "arg1type=Loc_and_arg2type=Org"
  	// arg_subtype: similar, for entity subtypes
  	if (usingFeature(types, checklist, "arg_type")) {
  		features.add("arg1type=" + arg0Type + "_and_arg2type=" + arg1.getType());
  	}
  	if (usingFeature(types,checklist,"arg_subtype")) {
  		features.add("arg1subtype="+arg0.getSubType()+"_and_arg2subtype="+arg1.getSubType());
  	}

  	// arg_order: which arg comes first in the sentence
  	if (usingFeature(types, checklist, "arg_order")) {
  		if (arg0.getSyntacticHeadTokenPosition() < arg1.getSyntacticHeadTokenPosition())
  			features.add("arg1BeforeArg2");
  	}
  	// same_head: whether the two args share the same syntactic head token
  	if (usingFeature(types, checklist, "same_head")) {
  	  if (arg0.getSyntacticHeadTokenPosition() == arg1.getSyntacticHeadTokenPosition())
  	    features.add("arguments_have_same_head");
  	}

  	// full_tree_path: Path from one arg to the other in the phrase structure tree,
  	// e.g., NNP -> PP -> NN <- NNP
  	if (usingFeature(types, checklist, "full_tree_path")) {
  	  //System.err.println("ARG0: " + arg0);
  	  //System.err.println("ARG0 HEAD: " + arg0.getSyntacticHeadTokenPosition());
  	  //System.err.println("TREE: " + tree);
  	  //System.err.println("SENTENCE: " + sentToString(arg0.getSentence()));
  		Tree arg0preterm = tree.getLeaves().get(arg0.getSyntacticHeadTokenPosition()).parent(tree);
  		Tree arg1preterm = tree.getLeaves().get(arg1.getSyntacticHeadTokenPosition()).parent(tree);
  		Tree join = tree.joinNode(arg0preterm, arg1preterm);
  		StringBuilder pathStringBuilder = new StringBuilder();
  		List<Tree> pathUp = join.dominationPath(arg0preterm);
  		Collections.reverse(pathUp);
  		for (Tree node : pathUp) {
  			if (node != join) {
  				pathStringBuilder.append(node.label().value() + " <- ");
  			}
  		}

  		for (Tree node : join.dominationPath(arg1preterm)) {
  			pathStringBuilder.append(((node == join) ? "" : " -> ") + node.label().value());
  		}
  		String pathString = pathStringBuilder.toString();
  		features.add(pathString);

  	}

  	int pathLength = tree.pathNodeToNode(tree.getLeaves().get(arg0.getSyntacticHeadTokenPosition()),
  			tree.getLeaves().get(arg1.getSyntacticHeadTokenPosition())).size();
  	// path_length: Length of the path in the phrase structure parse tree, integer-valued feature
  	if (usingFeature(types, checklist, "path_length")) {
  		// features.setCount("path_length", pathLength);
  	  throw new RuntimeException("ERROR: The path_length feature is not supported!");
  	}
  	// path_length_binary: Length of the path in the phrase structure parse tree, binary features
  	if (usingFeature(types, checklist, "path_length_binary")) {
  		features.add("path_length_" + pathLength);
  	}

  	/* entity_order
  	 * This tells you for each of the two args
  	 * whether there are other entities before or after that arg.
  	 * In particular, it can tell whether an arg is the first entity of its type in the sentence
  	 * (which can be useful for example for telling the gameWinner and gameLoser in NFL).
  	 * TODO: restrict this feature so that it only looks for
  	 * entities of the same type?
  	 * */
  	if (usingFeature(types, checklist, "entity_order")) {
  		String feature;
  		for (int i = 0; i < rel.getArgs().size(); i++) {
  		  // We already checked the class of the args at the beginning of the method
  		  EntityMention arg = (EntityMention) rel.getArgs().get(i);
  			for (EntityMention otherArg : rel.getSentence().get(MachineReadingAnnotations.EntityMentionsAnnotation.class)) {
  				if (otherArg.getSyntacticHeadTokenPosition() > arg.getSyntacticHeadTokenPosition()) {
  					feature = "arg" + i + "_before_" + otherArg.getType();
  					features.add(feature);
  				}
  				if (otherArg.getSyntacticHeadTokenPosition() < arg.getSyntacticHeadTokenPosition()) {
  					feature = "arg" + i + "_after_" + otherArg.getType();
  					features.add(feature);
  				}
  			}
  		}
  	}

  	// surface_distance: Number of tokens in the sentence between the two words, integer-valued feature
  	int surfaceDistance = Math.abs(arg0.getSyntacticHeadTokenPosition() - arg1.getSyntacticHeadTokenPosition());
  	if (usingFeature(types, checklist, "surface_distance")) {
  	  // features.setCount("surface_distance", surfaceDistance);
  	  throw new RuntimeException("ERROR: The feature surface_distance is not supported!");
  	}
  	// surface_distance_binary: Number of tokens in the sentence between the two words, binary features
  	if (usingFeature(types, checklist, "surface_distance_binary")) {
  		features.add("surface_distance_" + surfaceDistance);
  	}
  	// surface_distance_bins: number of tokens between the two args, binned to several intervals
  	if(usingFeature(types, checklist, "surface_distance_bins")) {
  	  if(surfaceDistance < 4){
  	    features.add("surface_distance_bin" + surfaceDistance);
  	  } else if(surfaceDistance < 6){
  	    features.add("surface_distance_bin_lt6");
  	  } else if(surfaceDistance < 10) {
  	    features.add("surface_distance_bin_lt10");
  	  } else {
  	    features.add("surface_distance_bin_ge10");
  	  }
  	}

  	// separate_surface_windows: windows of 1,2,3 tokens before and after args, for each arg separately
  	// Separate features are generated for windows to the left and to the right of the args.
  	// Features are concatenations of words in the window (or NULL for sentence boundary).
  	//
  	// conjunction_surface_windows: concatenation of the windows of the two args
  	//
  	// separate_surface_windows_POS: windows of POS tags of size 1,2,3 for each arg
  	//
  	// conjunction_surface_windows_POS: concatenation of windows of the args

  	List<EntityMention> args = new ArrayList<EntityMention>();
  	args.add(arg0); args.add(arg1);
  	for (int windowSize = 1; windowSize <= 3; windowSize++) {

  		String[] leftWindow, rightWindow, leftWindowPOS, rightWindowPOS;
  		leftWindow = new String[2];
  		rightWindow = new String[2];
  		leftWindowPOS = new String[2];
  		rightWindowPOS = new String[2];

  		for (int argn = 0; argn <= 1; argn++) {
  			int ind = args.get(argn).getSyntacticHeadTokenPosition();
  			for (int winnum = 1; winnum <= windowSize; winnum++) {
  				int windex = ind - winnum;
  				if (windex > 0) {
  					leftWindow[argn] = leaves.get(windex).label().value() + "_" + leftWindow[argn];
  					leftWindowPOS[argn] = leaves.get(windex).parent(tree).label().value() + "_" + leftWindowPOS[argn];
  				} else {
  					leftWindow[argn] = "NULL_" + leftWindow[argn];
  					leftWindowPOS[argn] = "NULL_" + leftWindowPOS[argn];
  				}
  				windex = ind + winnum;
  				if (windex < leaves.size()) {
  					rightWindow[argn] = rightWindow[argn] + "_" + leaves.get(windex).label().value();
  					rightWindowPOS[argn] = rightWindowPOS[argn] + "_" + leaves.get(windex).parent(tree).label().value();
  				} else {
  					rightWindow[argn] = rightWindow[argn] + "_NULL";
  					rightWindowPOS[argn] = rightWindowPOS[argn] + "_NULL";
  				}
  			}
  			if (usingFeature(types, checklist, "separate_surface_windows")) {
  				features.add("left_window_"+windowSize+"_arg_" + argn + ": " + leftWindow[argn]);
  				features.add("left_window_"+windowSize+"_POS_arg_" + argn + ": " + leftWindowPOS[argn]);
  			}
  			if (usingFeature(types, checklist, "separate_surface_windows_POS")) {
  				features.add("right_window_"+windowSize+"_arg_" + argn + ": " + rightWindow[argn]);
  				features.add("right_window_"+windowSize+"_POS_arg_" + argn + ": " + rightWindowPOS[argn]);
  			}

  		}
  		if (usingFeature(types, checklist, "conjunction_surface_windows")) {
  			features.add("left_windows_"+windowSize+": " + leftWindow[0] + "__" + leftWindow[1]);
  			features.add("right_windows_"+windowSize+": " + rightWindow[0] + "__" + rightWindow[1]);
  		}
  		if (usingFeature(types, checklist, "conjunction_surface_windows_POS")) {
  			features.add("left_windows_"+windowSize+"_POS: " + leftWindowPOS[0] + "__" + leftWindowPOS[1]);
  			features.add("right_windows_"+windowSize+"_POS: " + rightWindowPOS[0] + "__" + rightWindowPOS[1]);
  		}
  	}

  	// arg_words:  The actual arg tokens as separate features, and concatenated
  	String word0 = leaves.get(arg0.getSyntacticHeadTokenPosition()).label().value();
  	String word1 = leaves.get(arg1.getSyntacticHeadTokenPosition()).label().value();
  	if (usingFeature(types, checklist, "arg_words")) {
  	  if(doNotLexicalizeFirstArg == false)
  	    features.add("word_arg0: " + word0);
  		features.add("word_arg1: " + word1);
  		if(doNotLexicalizeFirstArg == false)
  		  features.add("words: " + word0 + "__" + word1);
  	}

  	// arg_POS:  POS tags of the args, as separate features and concatenated
  	String pos0 = leaves.get(arg0.getSyntacticHeadTokenPosition()).parent(tree).label().value();
  	String pos1 = leaves.get(arg1.getSyntacticHeadTokenPosition()).parent(tree).label().value();
  	if (usingFeature(types, checklist, "arg_POS")) {
  		features.add("POS_arg0: " + pos0);
  		features.add("POS_arg1: " + pos1);
  		features.add("POSs: " + pos0 + "__" + pos1);
  	}

  	// adjacent_words: words immediately to the left and right of the args
  	if(usingFeature(types, checklist, "adjacent_words")){
  	  for(int i = 0; i < rel.getArgs().size(); i ++){
  	    Span s = ((EntityMention) rel.getArg(i)).getHead();
  	    if(s.start() > 0){
  	      String v = tokens.get(s.start() - 1).word();
  	      features.add("leftarg" + i + "-" + v);
  	    }
  	    if(s.end() < tokens.size()){
  	      String v = tokens.get(s.end()).word();
  	      features.add("rightarg" + i + "-" + v);
  	    }
  	  }
  	}

  	// entities_between_args:  binary feature for each type specifying whether there is an entity of that type in the sentence
  	// between the two args.
  	// e.g. "entity_between_args: Loc" means there is at least one entity of type Loc between the two args
  	if (usingFeature(types, checklist, "entities_between_args")) {
  		for (EntityMention arg : rel.getSentence().get(MachineReadingAnnotations.EntityMentionsAnnotation.class)) {
  			if ((arg.getSyntacticHeadTokenPosition() > arg0.getSyntacticHeadTokenPosition() && arg.getSyntacticHeadTokenPosition() < arg1.getSyntacticHeadTokenPosition())
  					|| (arg.getSyntacticHeadTokenPosition() > arg1.getSyntacticHeadTokenPosition() && arg.getSyntacticHeadTokenPosition() < arg0.getSyntacticHeadTokenPosition())) {
  				features.add("entity_between_args: " + arg.getType());
  			}
  		}
  	}

  	// entity_counts: For each type, the total number of entities of that type in the sentence (integer-valued feature)
  	// entity_counts_binary: Counts of entity types as binary features.
  	Counter<String> typeCounts = new ClassicCounter<String>();
  	for (EntityMention arg : rel.getSentence().get(MachineReadingAnnotations.EntityMentionsAnnotation.class))
  	  typeCounts.incrementCount(arg.getType());
  	for (String type : typeCounts.keySet()) {
  	  if (usingFeature(types,checklist,"entity_counts")) {
  	    // features.add("entity_counts_"+type,typeCounts.getCount(type));
  	    throw new RuntimeException("ERROR: The feature entity_counts is not supported!");
  	  }
  	  if(usingFeature(types, checklist, "entity_counts_bins")){
  	    double typeCount = typeCounts.getCount(type);
  	    if(typeCount < 4){
          features.add("entity_counts_bin" + (int) typeCount);
        } else if(typeCount < 6){
          features.add("entity_counts_bin_lt6");
        } else if(typeCount < 10) {
          features.add("entity_counts_bin_lt10");
        } else {
          features.add("entity_counts_bin_ge10");
        }
  	  }
  	  if (usingFeature(types,checklist,"entity_counts_binary"))
  	    features.add("entity_counts_"+type+": "+typeCounts.getCount(type));
  	}

  	// surface_path: concatenation of tokens between the two args
  	// surface_path_POS: concatenation of POS tags between the args
  	// surface_path_selective: concatenation of tokens between the args which are nouns or verbs
  	StringBuilder sb = new StringBuilder();
  	StringBuilder sbPOS = new StringBuilder();
  	StringBuilder sbSelective = new StringBuilder();
  	for (int i = Math.min(arg0.getSyntacticHeadTokenPosition(), arg1.getSyntacticHeadTokenPosition()) + 1; i < Math.max(arg0.getSyntacticHeadTokenPosition(), arg1.getSyntacticHeadTokenPosition()); i++) {
  		String word = leaves.get(i).label().value();
  		sb.append(word + "_");
  		String pos = leaves.get(i).parent(tree).label().value();
  		sbPOS.append(pos + "_");
  		if (pos.equals("NN") || pos.equals("NNS") || pos.equals("NNP") || pos.equals("NNPS") || pos.equals("VB")
  				|| pos.equals("VBN") || pos.equals("VBD") || pos.equals("VBG") || pos.equals("VBP") || pos.equals("VBZ")) {
  			sbSelective.append(word + "_");
  		}
  	}
  	if (usingFeature(types, checklist, "surface_path")) {
  		features.add("surface_path: " + sb);
  	}
  	if (usingFeature(types, checklist, "surface_path_POS")) {
  		features.add("surface_path_POS: " + sbPOS);
  	}
  	if (usingFeature(types, checklist, "surface_path_selective")) {
  		features.add("surface_path_selective: " + sbSelective);
  	}

    int swStart = -1, swEnd = -1;
    if (arg0.getSyntacticHeadTokenPosition() < arg1.getSyntacticHeadTokenPosition()){
      swStart = arg0.getExtentTokenEnd();
      swEnd = arg1.getExtentTokenStart();
    } else {
      swStart = arg1.getExtentTokenEnd();
      swEnd = arg0.getExtentTokenStart();
    }

  	// span_words_unigrams: words that appear in between the two arguments
    if (usingFeature(types, checklist, "span_words_unigrams")) {
      for(int i = swStart; i < swEnd; i ++){
        features.add("span_word:" + tokens.get(i).word());
      }
  	}

    // span_words_bigrams: bigrams of words that appear in between the two arguments
    if (usingFeature(types, checklist, "span_words_bigrams")) {
      for(int i = swStart; i < swEnd - 1; i ++){
        features.add("span_bigram:" + tokens.get(i).word() + "-" + tokens.get(i + 1).word());
      }
    }

  	if (usingFeature(types, checklist, "span_words_trigger")) {
  	  for (int i = swStart; i < swEnd; i++) {
  	    String trigger = tokens.get(i).get(TriggerAnnotation.class);
  	    if (trigger != null && trigger.startsWith("B-"))
  	      features.add("span_words_trigger=" + trigger.substring(2));
  	  }
  	}

    if (usingFeature(types, checklist, "arg2_number")) {
      if (arg1.getType().equals("NUMBER")){
        try {
          int value = Integer.parseInt(arg1.getValue());

          if (2 <= value && value <= 100)
            features.add("arg2_number");
          if (2 <= value && value <= 19)
            features.add("arg2_number_2");
          if (20 <= value && value <= 59)
            features.add("arg2_number_20");
          if (60 <= value && value <= 100)
            features.add("arg2_number_60");
          if (value >= 100)
            features.add("arg2_number_100");
        } catch (NumberFormatException e) {}
      }
    }

    if (usingFeature(types, checklist, "arg2_date")) {
      if (arg1.getType().equals("DATE")){
        try {
          int value = Integer.parseInt(arg1.getValue());

          if (0 <= value && value <= 2010)
            features.add("arg2_date");
          if (0 <= value && value <= 999)
            features.add("arg2_date_0");
          if (1000 <= value && value <= 1599)
            features.add("arg2_date_1000");
          if (1600 <= value && value <= 1799)
            features.add("arg2_date_1600");
          if (1800 <= value && value <= 1899)
            features.add("arg2_date_1800");
          if (1900 <= value && value <= 1999)
            features.add("arg2_date_1900");
          if (value >= 2000)
            features.add("arg2_date_2000");
        } catch (NumberFormatException e) {}
      }
    }

    if (usingFeature(types, checklist, "arg_gender")) {
      boolean arg0Male = false, arg0Female = false;
      boolean arg1Male = false, arg1Female = false;
      System.out.println("Adding gender annotations!");

      int index = arg0.getExtentTokenStart();
      String gender = tokens.get(index).get(GenderAnnotation.class);
      System.out.println(tokens.get(index).word() + " -- " + gender);
      if (gender.equals("MALE"))
        arg0Male = true;
      else if (gender.equals("FEMALE"))
        arg0Female = true;

      index = arg1.getExtentTokenStart();
      gender = tokens.get(index).get(GenderAnnotation.class);
      if (gender.equals("MALE"))
        arg1Male = true;
      else if (gender.equals("FEMALE"))
        arg1Female = true;

      if (arg0Male) features.add("arg1_male");
      if (arg0Female) features.add("arg1_female");
      if (arg1Male) features.add("arg2_male");
      if (arg1Female) features.add("arg2_female");

      if ((arg0Male && arg1Male) || (arg0Female && arg1Female))
        features.add("arg_same_gender");
      if ((arg0Male && arg1Female) || (arg0Female && arg1Male))
        features.add("arg_different_gender");
    }

    List<String> tempDepFeatures = new ArrayList<String>(dependencyFeatures);
    if (tempDepFeatures.removeAll(types) || types.contains("all")) { // dependencyFeatures contains at least one of the features listed in types
      addDependencyPathFeatures(features, rel, arg0, arg1, types, checklist);
    }

  	if (!checklist.isEmpty() && !checklist.contains("all"))
  	  throw new AssertionError("FeatureFactory: features not handled: "+checklist);

    return true;
  }

  /**
   * Detects the true entity type based on the relation type
   * @param orgEntType
   * @param relType
   */
  private static String findTrueEntityType(String origEntType, String relType) {
    // this happens because "Infobox musical artist" contains slots for both PERSON and ORGANIZATION
    // Note: the strings in the comparisons below must be from Utils.entityTypeToString()!
    if(relType.startsWith("org:") && origEntType.contains("PERSON")) {
      String type = Utils.makeEntityType(EntityType.ORGANIZATION);
      Log.severe("TYPE CHANGE: from " + origEntType + " to " + type);
      return type;
    }

    else if(relType.startsWith("per:") && origEntType.contains("ORGANIZATION")) {
      String type = Utils.makeEntityType(EntityType.PERSON);
      Log.severe("TYPE CHANGE: from " + origEntType + " to " + type);
      return type;
    }

    return origEntType;
  }

  String sentToString(CoreMap sentence) {
    StringBuffer os = new StringBuffer();
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    if(tokens != null){
      boolean first = true;
      for(CoreLabel token: tokens) {
        if(! first) os.append(" ");
        os.append(token.word());
        first = false;
      }
    }

    return os.toString();
  }

  @SuppressWarnings("deprecation") // needed for MachineReadingAnnotations.DependencyAnnotation
  protected void addDependencyPathFeatures(
      Collection<String> features, RelationMention rel, EntityMention arg0, EntityMention arg1, List<String> types, List<String> checklist) {
    SemanticGraph graph = null;
    if(Constants.USE_OLD_CACHING) {
      // this works just for the cached sentences of KBP 2010 that do not come from the web
      graph = rel.getSentence().get(MachineReadingAnnotations.DependencyAnnotation.class);
    }
    if(graph == null){
      if(dependencyType == DEPENDENCY_TYPE.COLLAPSED_CCPROCESSED)
        graph = rel.getSentence().get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);
      else if(dependencyType == DEPENDENCY_TYPE.COLLAPSED)
        graph = rel.getSentence().get(SemanticGraphCoreAnnotations.CollapsedDependenciesAnnotation.class);
      else if(dependencyType == DEPENDENCY_TYPE.BASIC)
        graph = rel.getSentence().get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
      else
        throw new RuntimeException("ERROR: unknown dependency type: " + dependencyType);
    }
  	if (graph == null){
  	  logger.severe("Cannot find dependency graph for dependencyType " + dependencyType + " in the sentence below.");
      logger.severe("Current sentence from index " + rel.getSentence().get(SourceIndexAnnotation.class) + " is: " + Utils.sentenceToString(rel.getSentence(), true, true, true, true, true, true, false));
  	  assert(false);
  	} else {
  	  logger.fine("Dependency graph: " + graph);
  	}

  	IndexedWord node0 = graph.getNodeByIndexSafe(arg0.getSyntacticHeadTokenPosition() + 1);
  	IndexedWord node1 = graph.getNodeByIndexSafe(arg1.getSyntacticHeadTokenPosition() + 1);
  	if (node0 == null) {
  	  checklist.removeAll(dependencyFeatures);
  		return;
  	}
  	if (node1 == null) {
  	  checklist.removeAll(dependencyFeatures);
  		return;
  	}

  	List<SemanticGraphEdge> edgePath = graph.getShortestUndirectedPathEdges(node0, node1);
  	List<IndexedWord> pathNodes = graph.getShortestUndirectedPathNodes(node0, node1);

  	if (edgePath == null) {
  	  checklist.removeAll(dependencyFeatures);
  		return;
  	}

  	if (pathNodes == null || pathNodes.size() <= 1) { // arguments have the same head.
  	  checklist.removeAll(dependencyFeatures);
  		return;
  	}

  	// dependency_path: Concatenation of relations in the path between the args in the dependency graph, including directions
  	// e.g. "subj->  <-prep_in  <-mod"
  	// dependency_path_lowlevel: Same but with finer-grained syntactic relations
  	// e.g. "nsubj->  <-prep_in  <-nn"
  	if (usingFeature(types, checklist, "dependency_path")) {
  		features.add(generalizedDependencyPath(edgePath, node0));
  	}
  	String pathDescription = dependencyPath(edgePath, node0);
    if (usingFeature(types, checklist, "dependency_path_lowlevel")) {
  		features.add(pathDescription);
  	}

  	if (usingFeature(types, checklist, "different_sentences") && pathDescription.contains(CoreMapCombiner.FAKE_ROOT_NAME)) {
      features.add("different_sentences");
    }

    List<String> pathLemmas = new ArrayList<String>();
    List<String> noArgPathLemmas = new ArrayList<String>();
  	// do not add to pathLemmas words that belong to one of the two args
  	Set<Integer> indecesToSkip = new HashSet<Integer>();
  	for(int i = arg0.getExtentTokenStart(); i < arg0.getExtentTokenEnd(); i ++) indecesToSkip.add(i + 1);
    for(int i = arg1.getExtentTokenStart(); i < arg1.getExtentTokenEnd(); i ++) indecesToSkip.add(i + 1);
  	for (IndexedWord node : pathNodes){
      pathLemmas.add(Morphology.lemmaStatic(node.value(), node.tag(), true));
  	  if(! indecesToSkip.contains(node.index()))
  	    noArgPathLemmas.add(Morphology.lemmaStatic(node.value(), node.tag(), true));
  	}


   	// Verb-based features
  	// These features were designed on the assumption that verbs are often trigger words
  	// (specifically with the "Kill" relation from Roth CONLL04 in mind)
  	// but they didn't end up boosting performance on Roth CONLL04, so they may not be necessary.
  	//
  	// dependency_paths_to_verb: for each verb in the dependency path,
  	// the path to the left of the (lemmatized) verb, to the right, and both, e.g.
  	// "subj-> be"
  	// "be  <-prep_in  <-mod"
  	// "subj->  be  <-prep_in  <-mod"
  	// (Higher level relations used as opposed to "lowlevel" finer grained relations)
  	if (usingFeature(types, checklist, "dependency_paths_to_verb")) {
  		for (IndexedWord node : pathNodes) {
  			if (node.tag().contains("VB")) {
  				if (node.equals(node0) || node.equals(node1)) {
  					continue;
  				}
  				String lemma = Morphology.lemmaStatic(node.value(), node.tag(), true);
  				String node1Path = generalizedDependencyPath(graph.getShortestUndirectedPathEdges(node, node1), node);
  				String node0Path = generalizedDependencyPath(graph.getShortestUndirectedPathEdges(node0, node), node0);
  				features.add(node0Path + " " + lemma);
  				features.add(lemma + " " + node1Path);
  				features.add(node0Path + " " + lemma + " " + node1Path);
  			}
  		}
  	}
  	// dependency_path_stubs_to_verb:
  	// For each verb in the dependency path,
  	// the verb concatenated with the first (high-level) relation in the path from arg0;
  	// the verb concatenated with the first relation in the path from arg1,
  	// and the verb concatenated with both relations.  E.g. (same arguments and sentence as example above)
  	// "stub: subj->  be"
  	// "stub: be  <-mod"
  	// "stub: subj->  be  <-mod"
  	if (usingFeature(types, checklist, "dependency_path_stubs_to_verb")) {
  		for (IndexedWord node : pathNodes) {
  			SemanticGraphEdge edge0 = edgePath.get(0);
  			SemanticGraphEdge edge1 = edgePath.get(edgePath.size() - 1);
  			if (node.tag().contains("VB")) {
  				if (node.equals(node0) || node.equals(node1)) {
  					continue;
  				}
  				String lemma = Morphology.lemmaStatic(node.value(), node.tag(), true);
  				String edge0str, edge1str;
  				if (node0.equals(edge0.getGovernor())) {
  					edge0str = "<-" + generalizeRelation(edge0.getRelation());
  				} else {
  					edge0str = generalizeRelation(edge0.getRelation()) + "->";
  				}
  				if (node1.equals(edge1.getGovernor())) {
  					edge1str = generalizeRelation(edge1.getRelation()) + "->";
  				} else {
  					edge1str = "<-" + generalizeRelation(edge1.getRelation());
  				}
  				features.add("stub: " + edge0str + " " + lemma);
  				features.add("stub: " + lemma + edge1str);
  				features.add("stub: " + edge0str + " " + lemma + " " + edge1str);
  			}
  		}
  	}

  	if (usingFeature(types, checklist, "verb_in_dependency_path")) {
  		for (IndexedWord node : pathNodes) {
  			if (node.tag().contains("VB")) {
  				if (node.equals(node0) || node.equals(node1)) {
  					continue;
  				}
  				SemanticGraphEdge rightEdge = graph.getShortestUndirectedPathEdges(node, node1).get(0);
  				SemanticGraphEdge leftEdge = graph.getShortestUndirectedPathEdges(node, node0).get(0);
  				String rightRelation, leftRelation;
  				boolean governsLeft = false, governsRight = false;
  				if (node.equals(rightEdge.getGovernor())) {
  					rightRelation = " <-" + generalizeRelation(rightEdge.getRelation());
  					governsRight = true;
  				} else {
  					rightRelation = generalizeRelation(rightEdge.getRelation()) + "-> ";
  				}
  				if (node.equals(leftEdge.getGovernor())) {
  					leftRelation = generalizeRelation(leftEdge.getRelation()) + "-> ";
  					governsLeft = true;
  				} else {
  					leftRelation = " <-" + generalizeRelation(leftEdge.getRelation());
  				}
  				String lemma = Morphology.lemmaStatic(node.value(), node.tag(), true);

  				if (governsLeft || governsRight) {
  				}
  				if (governsLeft) {
  					features.add("verb: " + leftRelation + lemma);
  				}
  				if (governsRight) {
  					features.add("verb: " + lemma + rightRelation);
  				}
  				if (governsLeft && governsRight) {
  					features.add("verb: " + leftRelation + lemma + rightRelation);
  				}
  			}
  		}
  	}


  	// FEATURES FROM BJORNE ET AL., BIONLP'09
  	// dependency_path_words: generates a feature for each word in the dependency path (lemmatized)
  	// dependency_path_POS_unigrams: generates a feature for the POS tag of each word in the dependency path
  	if (usingFeature(types, checklist, "dependency_path_words")) {
  	  for (String lemma : noArgPathLemmas)
  	    features.add("word_in_dependency_path:" + lemma);
  	}
  	if (usingFeature(types, checklist, "dependency_path_POS_unigrams")) {
  	  for (IndexedWord node : pathNodes)
  	    if (!node.equals(node0) && !node.equals(node1))
  	      features.add("POS_in_dependency_path: "+node.tag());
  	}

  	// dependency_path_word_n_grams: n-grams of words (lemmatized) in the dependency path, n=2,3,4
  	// dependency_path_POS_n_grams: n-grams of POS tags of words in the dependency path, n=2,3,4
  	for (int node = 0; node < pathNodes.size(); node++) {
  	  for (int n = 2; n <= 4; n++) {
  	    if (node+n > pathNodes.size())
  	      break;
  	    StringBuilder sb = new StringBuilder();
  	    StringBuilder sbPOS = new StringBuilder();

  	    for (int elt = node; elt < node+n; elt++) {
  	      sb.append(pathLemmas.get(elt));
  	      sb.append("_");
  	      sbPOS.append(pathNodes.get(elt).tag());
  	      sbPOS.append("_");
  	    }
  	    if (usingFeature(types, checklist, "dependency_path_word_n_grams"))
          features.add("dependency_path_"+n+"-gram: "+sb);
        if (usingFeature(types,checklist, "dependency_path_POS_n_grams"))
          features.add("dependency_path_POS_"+n+"-gram: "+sbPOS);
  	  }
  	}
  	// dependency_path_edge_n_grams: n_grams of relations (high-level) in the dependency path, undirected, n=2,3,4
  	// e.g. "subj -- prep_in -- mod"
  	// dependency_path_edge_lowlevel_n_grams: similar, for fine-grained relations
  	//
  	// dependency_path_node-edge-node-grams: trigrams consisting of adjacent words (lemmatized) in the dependency path
  	// and the relation between them (undirected)
  	// dependency_path_node-edge-node-grams_lowlevel: same, using fine-grained relations
  	//
  	// dependency_path_edge-node-edge-grams: trigrams consisting of words (lemmatized) in the dependency path
  	// and the incoming and outgoing relations (undirected)
  	// e.g. "subj -- television -- mod"
  	// dependency_path_edge-node-edge-grams_lowlevel: same, using fine-grained relations
  	//
  	// dependency_path_directed_bigrams: consecutive words in the dependency path (lemmatized) and the direction
  	// of the dependency between them
  	// e.g. "Theatre -> exhibit"
  	//
  	// dependency_path_edge_unigrams: feature for each (fine-grained) relation in the dependency path,
  	// with its direction in the path and whether it's at the left end, right end, or interior of the path.
  	// e.g. "prep_at ->  - leftmost"
  	for (int edge = 0; edge < edgePath.size(); edge++) {
  	  if (usingFeature(types, checklist, "dependency_path_edge_n_grams") ||
  	      usingFeature(types, checklist, "dependency_path_edge_lowlevel_n_grams")) {
  	    for (int n = 2; n <= 4; n++) {
  	      if (edge+n > edgePath.size())
  	        break;
  	      StringBuilder sbRelsHi = new StringBuilder();
  	      StringBuilder sbRelsLo = new StringBuilder();
  	      for (int elt = edge; elt < edge+n; elt++) {
  	        GrammaticalRelation gr = edgePath.get(elt).getRelation();
  	        sbRelsHi.append(generalizeRelation(gr));
  	        sbRelsHi.append("_");
  	        sbRelsLo.append(gr);
  	        sbRelsLo.append("_");
  	      }
  	      if (usingFeature(types, checklist, "dependency_path_edge_n_grams"))
  	        features.add("dependency_path_edge_"+n+"-gram: "+sbRelsHi);
  	      if (usingFeature(types, checklist, "dependency_path_edge_lowlevel_n_grams"))
  	        features.add("dependency_path_edge_lowlevel_"+n+"-gram: "+sbRelsLo);
  	    }
  	  }
  	  if (usingFeature(types, checklist, "dependency_path_node-edge-node-grams"))
  	    features.add(
  	        "dependency_path_node-edge-node-gram: "+
  	        pathLemmas.get(edge)+" -- "+
  	        generalizeRelation(edgePath.get(edge).getRelation())+" -- "+
  	        pathLemmas.get(edge+1));
  	  if (usingFeature(types, checklist, "dependency_path_node-edge-node-grams_lowlevel"))
  	    features.add(
  	        "dependency_path_node-edge-node-gram_lowlevel: "+
  	        pathLemmas.get(edge)+" -- "+
  	        edgePath.get(edge).getRelation()+" -- "+
  	        pathLemmas.get(edge+1));
  	  if (usingFeature(types,checklist, "dependency_path_edge-node-edge-grams") && edge > 0)
  	    features.add(
  	        "dependency_path_edge-node-edge-gram: "+
            generalizeRelation(edgePath.get(edge-1).getRelation())+" -- "+
            pathLemmas.get(edge)+" -- "+
            generalizeRelation(edgePath.get(edge).getRelation()));
  	  if (usingFeature(types,checklist,"dependency_path_edge-node-edge-grams_lowlevel") && edge > 0)
  	    features.add(
  	        "dependency_path_edge-node-edge-gram_lowlevel: "+
            edgePath.get(edge-1).getRelation()+" -- "+
            pathLemmas.get(edge)+" -- "+
            edgePath.get(edge).getRelation());
  	  String dir = pathNodes.get(edge).equals(edgePath.get(edge).getDependent()) ? " -> " : " <- ";
  	  if (usingFeature(types, checklist, "dependency_path_directed_bigrams"))
  	    features.add(
  	        "dependency_path_directed_bigram: "+
  	        pathLemmas.get(edge)+
  	        dir+
  	        pathLemmas.get(edge+1));
  	  if (usingFeature(types, checklist, "dependency_path_edge_unigrams"))
  	    features.add(
  	        "dependency_path_edge_unigram: "+
  	        edgePath.get(edge).getRelation() +
  	        dir+
  	        (edge==0 ? " - leftmost" : edge==edgePath.size()-1 ? " - rightmost" : " - interior"));
  	}

  	// dependency_path_length: number of edges in the path between args in the dependency graph, integer-valued
  	// dependency_path_length_binary: same, as binary features
  	if (usingFeature(types, checklist, "dependency_path_length")) {
  		// features.add("dependency_path_length", edgePath.size());
  	  throw new RuntimeException("ERROR: The feature dependency_path_length is not supported!");
  	}
  	if (usingFeature(types, checklist, "dependency_path_length_binary")) {
  		features.add("dependency_path_length_" + new DecimalFormat("00").format(edgePath.size()));
  	}

  	 if (usingFeature(types, checklist, "dependency_path_trigger")) {
  	   List<CoreLabel> tokens = rel.getSentence().get(TokensAnnotation.class);

       for (IndexedWord node : pathNodes) {
         int index = node.index();
         if (indecesToSkip.contains(index)) continue;

         String trigger = tokens.get(index - 1).get(TriggerAnnotation.class);
         if (trigger != null && trigger.startsWith("B-"))
           features.add("dependency_path_trigger=" + trigger.substring(2));
       }
     }
  }

  /**
   * Helper method that checks if a feature type "type" is present in the list of features "types"
   * and removes it from "checklist"
   * @param types
   * @param checklist
   * @param type
   * @return true if types contains type
   */
  private boolean usingFeature(final List<String> types, List<String> checklist, String type) {
    checklist.remove(type);
    return types.contains(type) || types.contains("all");
  }

  private static GrammaticalRelation generalizeRelation(GrammaticalRelation gr) {
    final GrammaticalRelation[] GENERAL_RELATIONS = new GrammaticalRelation[] { EnglishGrammaticalRelations.SUBJECT,
        EnglishGrammaticalRelations.COMPLEMENT, EnglishGrammaticalRelations.CONJUNCT,
        EnglishGrammaticalRelations.MODIFIER, };
    for (GrammaticalRelation generalGR : GENERAL_RELATIONS) {
      if (generalGR.isAncestor(gr)) {
        return generalGR;
      }
    }
    if (gr.equals(EnglishGrammaticalRelations.CONTROLLING_SUBJECT)) {
      return EnglishGrammaticalRelations.SUBJECT;
    }
    return gr;
  }

  /*
   * Under construction
   */

  public static List<String> dependencyPathAsList(List<SemanticGraphEdge> edgePath, IndexedWord node, boolean generalize) {
    if(edgePath == null) return null;
    List<String> path = new ArrayList<String>();
    for (SemanticGraphEdge edge : edgePath) {
      IndexedWord nextNode;
      GrammaticalRelation relation;
      if (generalize) {
        relation = generalizeRelation(edge.getRelation());
      } else {
        relation = edge.getRelation();
      }

      if (node.equals(edge.getDependent())) {
        String v = (relation + "->").intern();
        path.add(v);
        nextNode = edge.getGovernor();
      } else {
        String v = ("<-" + relation).intern();
        path.add(v);
        nextNode = edge.getDependent();
      }
      node = nextNode;
    }

    return path;
  }

  public static String dependencyPath(List<SemanticGraphEdge> edgePath, IndexedWord node) {
    // the extra spaces are to maintain compatibility with existing relation extraction models
    return " " + StringUtils.join(dependencyPathAsList(edgePath, node, false), "  ") + " ";
  }

  public static String generalizedDependencyPath(List<SemanticGraphEdge> edgePath, IndexedWord node) {
    // the extra spaces are to maintain compatibility with existing relation extraction models
    return " " + StringUtils.join(dependencyPathAsList(edgePath, node, true), "  ") + " ";
  }

}
