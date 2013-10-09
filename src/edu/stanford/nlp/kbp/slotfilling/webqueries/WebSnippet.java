package edu.stanford.nlp.kbp.slotfilling.webqueries;

/**
 * Represents a web snippet
 *
 * @author Angel Chang
 */
public class WebSnippet {
  int rank;     // 0 for unknown rank
  String link;  // null for unknown link
  String text;

  public WebSnippet(String text) {
    this.text = text;
  }

  public WebSnippet(int rank, String link, String text) {
    this.rank = rank;
    this.link = link;
    this.text = text;
  }

  public int getRank() {
    return rank;
  }

  public void setRank(int rank) {
    this.rank = rank;
  }

  public String getLink() {
    return link;
  }

  public void setLink(String link) {
    this.link = link;
  }

  public String getText() {
    return text;
  }

  public void setText(String text) {
    this.text = text;
  }

  public String toString(String delimiter)
  {
    StringBuilder sb = new StringBuilder();
    sb.append(rank).append(delimiter);
    sb.append(link).append(delimiter);
    sb.append(text);
    return sb.toString();
  }

  public String toString()
  {
    return toString(",");
  }
  
  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }

    WebSnippet that = (WebSnippet) o;

    if (rank != that.rank) {
      return false;
    }
    if (link != null ? !link.equals(that.link) : that.link != null) {
      return false;
    }
    if (text != null ? !text.equals(that.text) : that.text != null) {
      return false;
    }

    return true;
  }

  @Override
  public int hashCode() {
    int result = rank;
    result = 31 * result + (link != null ? link.hashCode() : 0);
    result = 31 * result + (text != null ? text.hashCode() : 0);
    return result;
  }
}
