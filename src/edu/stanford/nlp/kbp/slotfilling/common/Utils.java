package edu.stanford.nlp.kbp.slotfilling.common;

import java.io.PrintStream;
import java.io.StringReader;
import java.net.UnknownHostException;
import java.util.List;
import java.util.regex.Pattern;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreAnnotations.DocIDAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations.RelationMentionsAnnotation;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.DatetimeAnnotation;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.SlotMentionsAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.process.AbstractTokenizer;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;

public class Utils {
  /**
   * Prints a pipelined sentence in a format easy on human eyes
   * @param os
   * @param sentence
   */
  public static void printSentence(PrintStream os, CoreMap sentence) {
    os.print(sentenceToString(sentence));
  }

  public static String sentenceToString(CoreMap sentence) {
    return sentenceToString(sentence, true, true, true, true,
        true, true, true, false);
  }
  
  public static String sentenceToMinimalString(CoreMap sentence) {
    return sentenceToString(sentence, true, 
        false, false, false,
        false, false, false, false);
  }
  
  public static String sentenceToString(CoreMap sentence,
      boolean showText,
      boolean showTokens,
      boolean showEntities,
      boolean showSlots,
      boolean showRelations,
      boolean showDocId,
      boolean showDatetime) {
    return sentenceToString(sentence, showText, showTokens, showEntities, showSlots, showRelations, showDocId, showDatetime, false);
  }

  public static String sentenceToString(CoreMap sentence,
      boolean showText,
      boolean showTokens,
      boolean showEntities,
      boolean showSlots,
      boolean showRelations,
      boolean showDocId,
      boolean showDatetime,
      boolean showParseTree) {
    StringBuffer os = new StringBuffer();
    boolean justText = showText && ! showTokens && ! showEntities && ! showSlots && ! showRelations && ! showDocId && ! showDatetime;

    //
    // Print text and tokens
    //
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    if(tokens != null){
      if(showText){
        if(! justText) os.append("TEXT: ");
        boolean first = true;
        for(CoreLabel token: tokens) {
          if(! first) os.append(" ");
          os.append(token.word());
          first = false;
        }
        if(! justText) os.append("\n");
      }
      if(showTokens) {
        os.append("TOKENS:");
        String[] tokenAnnotations = new String[] {
            "Text", "PartOfSpeech", "NamedEntityTag" , "Antecedent"
        };
        for (CoreLabel token: tokens) {
          os.append(" " + token.toShorterString(tokenAnnotations));
        }
        os.append("\n");
      }
    }
    
    Tree tree = sentence.get(TreeAnnotation.class);
    if(tree != null && showParseTree) {
      os.append(tree.toString());
      os.append("\n");
    }

    //
    // Print EntityMentions
    //
    List<EntityMention> ents = sentence.get(EntityMentionsAnnotation.class);
    if (ents != null && showEntities) {
      os.append("ENTITY MENTIONS:\n");
      for(EntityMention e: ents){
        os.append("\t" + e.toString() + "\n");
      }
    }

    //
    // Print SlotMentions
    //
    List<EntityMention> slots = sentence.get(SlotMentionsAnnotation.class);
    if (slots != null && showSlots) {
      os.append("SLOT CANDIDATES:\n");
      for(EntityMention e: slots){
        os.append("\t" + e.toString() + "\n");
      }
    }

    //
    // Print RelationMentions
    //
    List<RelationMention> relations = sentence.get(RelationMentionsAnnotation.class);
    if(relations != null && showRelations){
      os.append("RELATION MENTIONS:\n");
      for(RelationMention r: relations){
        os.append(r.toString() + "\n");
      }
    }

    String docId = sentence.get(DocIDAnnotation.class);
    if (docId != null && showDocId) {
      os.append("DOCID: " + docId + "\n");
    }

    String datetime = sentence.get(DatetimeAnnotation.class);
    if (datetime != null && showDatetime) {
      os.append("DATETIME: " + datetime + "\n");
    }

    os.append("\n");
    return os.toString();
  }
  
  public static String makeEntityType(EntityType et) {
    return ("ENT:" + Utils.entityTypeToString(et));
  }

  public static String entityTypeToString(EntityType et) {
    switch (et) {
    case PERSON:
      return "PERSON";
    case ORGANIZATION:
      return "ORGANIZATION";
    default:
      throw new RuntimeException("Unknown EntityType " + et);
    }
  }

  /**
   * Tokenizes a string using our default tokenizer
   */
  public static List<CoreLabel> tokenize(String text) {
    AbstractTokenizer<CoreLabel> tokenizer;
    CoreLabelTokenFactory tokenFactory = new CoreLabelTokenFactory();
    // note: ptb3Escaping must be true because this is the default behavior in the pipeline
    // String options = "ptb3Escaping=false";
    String options="";
    StringReader sr = new StringReader(text);
    tokenizer = new PTBTokenizer<CoreLabel>(sr, tokenFactory, options);
    List<CoreLabel> tokens = tokenizer.tokenize();
    return tokens;
  }
  
  public static String [] tokenizeToStrings(String text) {
    List<CoreLabel> tokens = Utils.tokenize(text);
    String [] stringToks = new String[tokens.size()];
    for(int i = 0; i < tokens.size(); i ++) stringToks[i] = tokens.get(i).word();
    return stringToks;
  }

  /**
   * Verifies if the tokens in needle are contained in the tokens in haystack.
   * @param needle
   * @param haystack
   * @param caseInsensitive Perform case insensitive matching of token texts if true
   */
  public static boolean contained(List<CoreLabel> needle, List<CoreLabel> haystack, boolean caseInsensitive) {
    int index = findStartIndex(needle, haystack, caseInsensitive);
    return (index >= 0);
  }

  public static int findStartIndex(List<CoreLabel> needle, List<CoreLabel> haystack, boolean caseInsensitive) {
    return findStartIndex(needle, haystack, caseInsensitive, 0);
  }

  public static int findStartIndex(List<CoreLabel> needle, List<CoreLabel> haystack, boolean caseInsensitive, int searchStart) {
    for(int start = searchStart; start <= haystack.size() - needle.size(); start ++){
      boolean failed = false;
      for(int i = 0; i < needle.size(); i ++){
        String n = needle.get(i).word();
        String h = haystack.get(start + i).word();
        if(caseInsensitive && ! n.equalsIgnoreCase(h)){
          failed = true;
          break;
        }
        if(! caseInsensitive && ! n.equals(h)){
          failed = true;
          break;
        }
      }
      if(! failed) return start;
    }
    return -1;
  }

  static Pattern escaper = Pattern.compile("([^a-zA-z0-9])"); // should also include ,:; etc.?
  /**
   * Builds a new string where characters that have special meaning in Java regexes are escaped
   * @param s
   */
  public static String escapeSpecialRegexCharacters(String s) {
    return escaper.matcher(s).replaceAll("\\\\$1");
  }
  
  public static String sentenceSpanString(List<CoreLabel> tokens, Span span) {
    StringBuffer os = new StringBuffer();
    for(int i = span.start(); i < span.end(); i ++){
      if(i > span.start()) os.append(" ");
      os.append(tokens.get(i).word());
    }
    return os.toString();
  }
  
  /**
   * Replaces all instance of HOSTNAME with the actual hostname of this machine
   * @param value
   * @throws UnknownHostException
   */
  public static String convertToHostName(String value) throws UnknownHostException {
    String hn = getHostName();
    int firstDot = hn.indexOf('.');
    if(firstDot > 0) hn = hn.substring(0, firstDot);
    value = value.replaceAll("HOSTNAME", hn);
    return value;
  }
  private static String getHostName() throws UnknownHostException {
    java.net.InetAddress localMachine = java.net.InetAddress.getLocalHost();
    String hn = localMachine.getHostName();
    Log.severe("Hostname of local machine: " + hn);
    return hn;
  }

  public static void main(String[] args) throws Exception {
    String s = args[0];
    System.out.println(escapeSpecialRegexCharacters(s));
  }
}
