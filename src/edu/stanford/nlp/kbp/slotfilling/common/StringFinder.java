package edu.stanford.nlp.kbp.slotfilling.common;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

/**
 * This class builds a pattern out of a whole bunch of strings and
 * then matches either an input string or a tokenized sentence against
 * that set of strings.  The patterns are intended to be entity names,
 * so it has a few string manipulations that make the pattern better
 * able to detect the names in text, and there are regex command
 * sequences which aren't properly escaped.  The tokenized sentences
 * are expected to be of the kind produced by PTBTokenizer.
 * <br>
 * TODO: probably belongs elsewhere
 */
public class StringFinder {
  final Pattern regex;
  final String pattern;
  final Pattern justTheStringRegex;
  
  public StringFinder(Iterable<String> matches) {
    this(matches, false);
  }

  /**
   * Builds a regex that must match at least one of the match strings,
   * but can match it anywhere in the test string, and keeps a Pattern
   * containing that regex.
   */  
  public StringFinder(Iterable<String> matches, boolean caseInsensitive) {
    StringBuilder pattern = new StringBuilder();
    StringBuilder justTheStringPattern = new StringBuilder();
    pattern.append(".*(?:");
    justTheStringPattern.append("(?:");
    boolean firstPiece = true;
    for (String match : matches) {
      if (firstPiece) {
        firstPiece = false;
      } else {
        pattern.append("|");
        justTheStringPattern.append("|");
      }
      String cleanMatch = cleanMatchRegex(match);
      pattern.append(cleanMatch);
      justTheStringPattern.append(cleanMatch);
    }
    pattern.append(").*");
    justTheStringPattern.append(")");
    this.pattern = pattern.toString();
    // System.out.println("PATTERN: " + pattern);
    
    if(! caseInsensitive){
      this.regex = Pattern.compile(pattern.toString());
      this.justTheStringRegex = Pattern.compile(justTheStringPattern.toString());
    }
    else{
      this.regex = Pattern.compile(pattern.toString(), Pattern.CASE_INSENSITIVE);
      this.justTheStringRegex = Pattern.compile(justTheStringPattern.toString(), Pattern.CASE_INSENSITIVE);
    }
  }
  
  /**
   * Convenience method that calls the List constructor
   */
  public StringFinder(String ... matches) {
    this(Arrays.asList(matches));
  }

  /**
   * Returns true if the haystack contains any of the entity strings
   * given earlier.
   */
  public boolean matches(String haystack) {
    return regex.matcher(haystack).matches();
  }

  /**
   * Returns true if the sentence contains any of the entity strings
   * given earlier.  Skips commas.
   */
  public boolean matches(CoreMap sentence) {
    //System.out.println("Haystack:");
    //System.out.println(haystack);
    if (matches(toMatchString(sentence)))
      return true;
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    if (tokens == null)
      return false;
    for (CoreLabel token : tokens) {
      String antecedent = token.get(CoreAnnotations.AntecedentAnnotation.class);
      if (antecedent == null)
        continue;
      if (matches(antecedent))
        return true;
    }
    return false;
  }
  
  public List<Pair<Integer, Integer>> whereItMatches(CoreMap sentence) {
    List<Pair<Integer, Integer>> matches =
      whereItMatches(toMatchString(sentence));
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    if (tokens == null)
      return matches;
    for (CoreLabel token : tokens) {
      String antecedent = token.get(CoreAnnotations.AntecedentAnnotation.class);
      Integer begin = token.get(CharacterOffsetBeginAnnotation.class);
      Integer end = token.get(CharacterOffsetEndAnnotation.class);
      if (antecedent == null || begin == null || end == null)
        continue;
      if (matches(antecedent)) {
        matches.add(new Pair<Integer, Integer>(begin, end));
      }
    }
    Collections.sort(matches, new Comparator<Pair<Integer, Integer>>() {
        public int compare(Pair<Integer, Integer> p1, 
                           Pair<Integer, Integer> p2) {
          if (p1.first() != p2.first()) {
            return p1.first() - p2.first();
          } else {
            return p1.second() - p2.second();
          }
        }
      });
    return matches;
  }
  
  public List<Pair<Integer, Integer>> whereItMatches(String haystack) {
    Matcher m = justTheStringRegex.matcher(haystack);
    List<Pair<Integer, Integer>> matches = new ArrayList<Pair<Integer,Integer>>();
    while(m.find()){
      int start = m.start();
      int end = m.end();
      matches.add(new Pair<Integer, Integer>(start, end));
    }
    return matches;
  }

  public static final String cleanMatchRegex(String match) {
    // TODO: spaces around a dash or period?
    match = match.toLowerCase();
    match = match.replaceAll(",", " ");
    match = match.replaceAll("\\.", "[ .]?");
    match = match.replaceAll(" +", " ");
    match = match.replaceAll("-", "[ -]?");
    
    // TODO: more special characters?
    match = match.replaceAll("\\+", "\\\\+");
    match = match.replaceAll("\\*", "\\\\*");
    match = match.replaceAll("\\(", "\\\\s*-lrb-\\\\s*");
    match = match.replaceAll("\\)", "\\\\s*-rrb\\\\s*");

    match = match.trim();
    return match;
  }

  static public String toMatchString(CoreMap sentence) {
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    StringBuilder haystack = new StringBuilder();
    for (CoreLabel token : tokens) {
      if (token.word().equals(","))
        continue;
      if (haystack.length() > 0) {
        haystack.append(" ");
      }
      haystack.append(token.word().toLowerCase());
    }
    // System.out.println("MATCHING OVER [" + haystack.toString() + "]");
    return haystack.toString();
  }
  
  public String toString() {
    return pattern;
  }
}
