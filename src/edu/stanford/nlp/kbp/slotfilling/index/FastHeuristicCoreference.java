package edu.stanford.nlp.kbp.slotfilling.index;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.kbp.slotfilling.common.Utils;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.StringUtils;

public class FastHeuristicCoreference {
  private final static Pattern[] personRegexes;
  private final static Pattern[] organizationRegexes;
  private static Set<String> possessivePatterns = new HashSet<String>();
  
  // we won't perform substringReplacement on the first minimumIndexForSubstringReplacement tokens
  private static int minimumIndexForSubstringReplacement = 20;
  
  static {
    List<Pattern> personRegexesList = new ArrayList<Pattern>();
    List<Pattern> organizationRegexesList = new ArrayList<Pattern>();
    for (String regex : new String[] { "he", "she", "his", "her" }) {
      personRegexesList.add(Pattern.compile("\\b" + regex + "\\b",
          Pattern.CASE_INSENSITIVE));
    }
    for (String regex : new String[] { "it", "its" }) {
      organizationRegexesList.add(Pattern.compile("\\b" + regex + "\\b",
          Pattern.CASE_INSENSITIVE));
    }

    personRegexes = personRegexesList.toArray(new Pattern[0]);
    organizationRegexes = organizationRegexesList.toArray(new Pattern[0]);
    for (String regex : new String[] { "his", "her", "its" }) {
      possessivePatterns.add("\\b" + regex + "\\b");
    }
  }
  
  public static String fastHeuristicCorefReplacement(String name, String text, boolean nameIsPerson) {
    text = subsequenceReplacement(name, text, nameIsPerson, minimumIndexForSubstringReplacement);
    text = pronounReplacement(name, text, nameIsPerson);
    if (!nameIsPerson) {
      text = abbreviationReplacement(name, text);
    }
    return text;
  }

  /**
   * Replace pronouns in text with name.
   * 
   * @param name the name of the person or organization
   * @param text the text to perform the replacements on
   * @param nameIsPerson true if name is a person, false if name is an organization
   * @return (potentially) modified version of text with the pronouns replaced
   */
  private static String pronounReplacement(String name, String text, boolean nameIsPerson) {
    Pattern[] patterns;
    if (nameIsPerson) {
      patterns = personRegexes;
    } else {
      patterns = organizationRegexes;
    }

    for (Pattern regex : patterns) {
      String replacement = name;
      if (possessivePatterns.contains(regex.pattern())) {
        replacement = replacement + "'s";
      }
      Matcher matcher = regex.matcher(text);
      text = matcher.replaceAll(replacement);
    }
    return text;
  }
  
  /**
   * Replaces the longest subsequences of words from "name" in "text" with name.  For example:
   * If the text is "D A B E" and name is "A B C", it should return "D A B C E".
   * 
   * @param name the entity name
   * @param text the text to do the replacements in
   * @param nameIsPerson true if name is a person, false if name is an organization
   * @param minimumIndex the minimum index of a token in text to consider (tokenized via {@link Utils.tokenize}) for replacement
   * @return the text with subsequences of words from name replaced with name
   */
  private static String subsequenceReplacement(String name, String text, boolean nameIsPerson, int minimumIndex) {
    int threshold = 1;
    if (nameIsPerson) {
      threshold = 0;
    }
    
    List<CoreLabel> orderedTokensInName = Utils.tokenize(name); 
    Set<String> tokensInName = new HashSet<String>();
    for (CoreLabel token : orderedTokensInName) {
      tokensInName.add(token.word());
    }

    // tokens that have already had replacements done
    List<CoreLabel> processedTokens = new ArrayList<CoreLabel>();
    // a buffer of tokens which start with Capital Letters and are subsets of tokensInName
    List<CoreLabel> unprocessedTokens = new ArrayList<CoreLabel>();
    
    int index = 0;
    for (CoreLabel token : Utils.tokenize(text)) {
      String tokenValue = token.word();
      if (tokenValue.length() == 0) {
        continue;
      }
      if (index >= minimumIndex && StringUtils.isCapitalized(tokenValue) && tokensInName.contains(tokenValue)) {
        unprocessedTokens.add(token);
      } else {
        // we've completed a sequence of capitalized words from "name" and we flush unprocessedTokens
        if (unprocessedTokens.size() > 0) {
          if (unprocessedTokens.size() > threshold) { // need to have at least two otherwise we can get pretty strange output
            processedTokens.addAll(orderedTokensInName);
          } else {
            processedTokens.addAll(unprocessedTokens);
          }
          unprocessedTokens.clear();
        }
        processedTokens.add(token);
      }
      index++;
    }
    
    // flush unprocessedTokens again
    if (unprocessedTokens.size() > threshold) {
      processedTokens.addAll(orderedTokensInName);
    } else {
      processedTokens.addAll(unprocessedTokens);
    }
    
    return originalSpacingSensitiveJoin(processedTokens);
  }

  /*
   * This method isn't quite general purpose since in substringReplacement, we
   * reinsert old CoreLabels which have screwy offsets. Thus, this method has
   * hacks so we don't need to adjust those.
   */
  private static String originalSpacingSensitiveJoin(List<CoreLabel> processedTokens) {
    if (processedTokens.size() == 0) {
      return "";
    }
    
    CoreLabel lastToken = processedTokens.get(0);
    StringBuffer buffer = new StringBuffer(lastToken.word());
    
    for (int i = 1; i < processedTokens.size(); i++) {
      CoreLabel currentToken = processedTokens.get(i);
      // in case our offsets get screwed up, fall back to putting 1 space between tokens
      int numSpaces = currentToken.beginPosition() - lastToken.endPosition();
      if (numSpaces < 0) {
        numSpaces = 1;
      }
      // also, we have a maximum of one space between tokens
      numSpaces = Math.min(1, numSpaces);
      buffer.append(StringUtils.repeat(' ', numSpaces) + currentToken.word());
      lastToken = currentToken;
    }
    
    return buffer.toString();
  }

  /*
   * Replace abbreviations of "name" in "text" with "name". This considers only
   * capitalized letters in name and two possible abbreviations (with and
   * without periods)
   */
  private static String abbreviationReplacement(String name, String text) {
    List<Character> capitalizedLetters = new ArrayList<Character>();
    for (CoreLabel token : Utils.tokenize(name)) {
      String tokenValue = token.word();
      if (StringUtils.isCapitalized(tokenValue)) {
        capitalizedLetters.add(tokenValue.charAt(0));
      }
    }
    String abbrev = StringUtils.join(capitalizedLetters, "");
    String abbrevWithDots = StringUtils.join(capitalizedLetters, ".") + ".";
    text = text.replace(abbrev, name);
    text = text.replace(abbrevWithDots, name);

    return text;
  }
  
  public static void main(String [] argv) {
    String input = "Theodore Roberts the actor is not to be confused with author Theodore Goodridge " +
        "Roberts, 1877-1953, who wrote the \"The Harbor Master\". Please see . " + 
        "Theodore Roberts (October 8, 1861, San Francisco, California – December 14, " +
        "1928, Hollywood, California) was an American movie and stage actor. He was a " +
        "stage actor decades before becoming lovable old man in silents. On stage in the " +
        "1890s he acted with Fanny Davenport in her play called Gismonda (1894) and later " +
        "in The Bird of Paradise (1912) with actress Laurette Taylor. " +
        "He started his film career in the 1910s in Hollywood, and was often was " +
        "associated in the productions of Cecil B. DeMille. Theodore liked enchiladas.";
    String output = fastHeuristicCorefReplacement("Theodore Roberts", input, true);    
    
    /*
    input = "Edgar (Eddie) Smith (December 14, 1913 - January 2, 1994) was a starting pitcher " + 
        "in Major League Baseball who played for the Philadelphia Athletics " + 
        "(1936-1939[start]), Chicago White Sox";
    String output = fastHeuristicCorefReplacement("Eddie Smith", input, true);
    */    
    
    System.out.println("in : " + input);
    System.out.println("out: " + output);
  }
}
