package edu.stanford.nlp.kbp.slotfilling.webqueries;

import edu.stanford.nlp.util.EditDistance;

import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Masks out part of the web snippet
 *
 * @author Angel Chang
 */
public class WebSnippetMasker {
  private static final Pattern emPattern = Pattern.compile("<em>(.*?)</em>|<b>(.*?)</b>");
  private static final String emWord = "*EMWORD*";

  boolean maskMarkedWords = true;
  boolean markEntityFiller = true;

  public WebSnippetMasker()
  {
  }

  public WebSnippetMasker(Properties props)
  {
    init(props);
  }

  public void init(Properties props)
  {
    maskMarkedWords = Boolean.parseBoolean(props.getProperty("kbp.websnippets.maskMarkedWords", "true"));
    markEntityFiller = Boolean.parseBoolean(props.getProperty("kbp.websnippets.markEntityFiller", "true"));
  }

  public String getMaskedText(String text, String entity, String filler)
  {
    if (maskMarkedWords) {
      if (markEntityFiller) {
        text = maskEntityFiller(text, entity, filler);
      } else {
        text = maskMarked(text);
      }
    }
    return text;
  }

  public static String maskEntityFiller(String text, String entity, String filler)
  {
    EditDistance dist = new EditDistance();
    StringBuilder sb = new StringBuilder();
    Matcher matcher = emPattern.matcher(text);
    int segStart = 0;
    while (matcher.find()) {
      int matchStart = matcher.start();
      int matchEnd = matcher.end();
      if (matchStart > segStart) {
        String seg = text.substring(segStart, matchStart);
        sb.append(seg);
      }
      String matched = matcher.group(1);
      if (matched == null) {
        matched = matcher.group(2);
      }
      double entityEditDistance = dist.score(matched, entity);
      double fillerEditDistance = dist.score(matched, filler);
      if (entityEditDistance <= fillerEditDistance) {
        sb.append(" *ENTITY* ");
      } else {
        sb.append(" *FILLER* ");
      }
      segStart = matchEnd;
    }
    if (segStart < text.length()) {
      String seg = text.substring(segStart, text.length());
      sb.append(seg);
    }
    return sb.toString();
  }

  public static String maskMarked(String text)
  {
    return emPattern.matcher(text).replaceAll(emWord);
  }
}
