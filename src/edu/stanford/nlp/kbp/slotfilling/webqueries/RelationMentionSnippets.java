package edu.stanford.nlp.kbp.slotfilling.webqueries;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.kbp.slotfilling.common.KBPSlot;

public class RelationMentionSnippets implements Serializable {
  private static final long serialVersionUID = 1L;

  // EE (entity/entity queries) is snippet for entity filler (slotValue)
  // POS (positive queries) is snippet for entity keyword  filler (slotValue)
  // NEU (neutral queries) is snippet for entity keyword
  // NEG (negated queries) is snippet for entity keyword -filler (slotValue is negated)
  // NGA (strongly/all negated queries) is snippet for (entity one, minus each term in two, and keyword)
  // UNK is unknown (some new fancy query type we don't understand)
  public static enum QueryType {
    EE, NEG, POS, NEU, NGA, UNK
  };

  //KBPRelationMention mention;
  String entityName;
  String slotName;
  String slotValue;
  String keyword; // null or "" if EE
  QueryType queryType;
  String queryTypeName;
  String queryString;  // actual query string
  long totalResultsCount;  // total number of results (possibly extracted from resultsInfo field)
  String resultsInfo; // String describing number and time it took to retrieve results
  List<WebSnippet> snippets;

  private final static Set<String> unknownQueryTypes = new HashSet<String>();

  public RelationMentionSnippets(QueryType queryType, String entityName, String slotName, String slotValue) {
    this.entityName = entityName;
    this.slotName = slotName;
    this.slotValue = slotValue;
    this.queryType = queryType;
    this.queryTypeName = queryType.name();
    this.snippets = new ArrayList<WebSnippet>();
  }

  public RelationMentionSnippets(String queryTypeName, String entityName, String slotName, String slotValue) {
    this.entityName = entityName;
    this.slotName = slotName;
    this.slotValue = slotValue;
    setQueryType(queryTypeName);
    this.snippets = new ArrayList<WebSnippet>();
  }

  public RelationMentionSnippets(KBPSlot mention) {
    entityName = mention.entityName;
    slotName = mention.slotName;
    slotValue = mention.slotValue;
    this.snippets = new ArrayList<WebSnippet>();
  }

  protected void setQueryType(String queryTypeName)
  {
    this.queryTypeName = queryTypeName;
    try {
      this.queryType = QueryType.valueOf(queryTypeName);
    } catch (IllegalArgumentException ex) {
      this.queryType = QueryType.UNK;
      boolean warn = false;
      synchronized (RelationMentionSnippets.class) {
        if (!unknownQueryTypes.contains(queryTypeName)) {
          unknownQueryTypes.add(queryTypeName);
          warn = true;
        }
      }
      if (warn) {
        System.err.println("WARNING: Unknown query type " + queryType);
      }
    }
  }

  public void add(String snippetText) {
    snippets.add(new WebSnippet(snippetText));
  }

  public void add(WebSnippet snippet) {
    snippets.add(snippet);
  }

  public List<WebSnippet> getSnippets() {
    return snippets;
  }

  public String getKeyword() {
    return keyword;
  }

  public String getSlotValue() {
    return slotValue;
  }

  public String getSlotName() {
    return slotName;
  }

  public String getEntityName() {
    return entityName;
  }

  public QueryType getQueryType() {
    return queryType;
  }

  public String getQueryTypeName() {
    return queryTypeName;
  }

  public String getQueryString() {
    return queryString;
  }

  public long getTotalResultsCount() {
    return totalResultsCount;
  }

  /* Returns string describing the snippets */
  public String getHeaderF0() {
    StringBuilder sb = new StringBuilder();
    sb.append(slotName).append("\t");
    sb.append(entityName).append("\t");
    if (keyword != null) {
      sb.append(queryTypeName).append("\t");
      sb.append(keyword).append("\t");
      sb.append(slotValue);
      if (resultsInfo != null) {
        sb.append("\t").append(resultsInfo);
      }
    } else {
      sb.append(slotValue);
    }
    return sb.toString();
  }

  public String toString() {
    char delimiter = '\t';
    char eol = '\n';
    // First line
    StringBuilder sb = new StringBuilder();
    sb.append(slotName).append(delimiter);
    sb.append(entityName).append(delimiter);
    sb.append(slotValue).append(delimiter);
    sb.append(keyword).append(delimiter);
    sb.append(queryTypeName).append(eol);
    // Second line
    sb.append("0").append(delimiter);
    sb.append(totalResultsCount).append(delimiter);
    sb.append(queryString).append(eol);
    // Snippets
    for (WebSnippet snippet:snippets) {
      sb.append(snippet.getRank()).append(delimiter);
      sb.append(snippet.getLink()).append(delimiter);
      sb.append(snippet.getText()).append(eol);
    }
    return sb.toString();
  }
/*
  public Annotation toAnnotation() {
    List<CoreMap> sentences = new ArrayList(snippets.size());
    for (WebSnippet snippet : snippets) {
      sentences.add(new Annotation(snippet.text));
    }
    Annotation annotation = new Annotation("");
    annotation.set(CoreAnnotations.IDAnnotation.class, getHeaderF0());
    annotation.set(CoreAnnotations.SentencesAnnotation.class, sentences);
    return annotation;
  }
*/
}