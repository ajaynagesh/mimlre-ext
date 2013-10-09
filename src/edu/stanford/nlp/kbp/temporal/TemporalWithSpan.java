package edu.stanford.nlp.kbp.temporal;

import java.util.List;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.time.SUTime.IsoDate;


public class TemporalWithSpan {
  public IsoDate t1, t2, t3, t4;
  // it can be of size 1 to 4
  public List<Span> spans;

  public TemporalWithSpan(IsoDate t1, IsoDate t2, IsoDate t3, IsoDate t4, List<Span> spans) {
    this.t1 = t1;
    this.t2 = t2;
    this.t3 = t3;
    this.t4 = t4;
    this.spans = spans;
  }

  public void setSpans(List<Span> spans) {
    this.spans = spans;
  }

  public int leftMostTokenPosition() {
    assert(spans != null);
    int leftMost = Integer.MAX_VALUE;
    for(Span span: spans) {
      if(span.start() < leftMost) leftMost = span.start();
    }
    return leftMost;
  }

  public int rightMostTokenPosition() {
    assert(spans != null);
    int rightMost = Integer.MIN_VALUE;
    for(Span span: spans) {
      if(span.end() < rightMost) rightMost = span.end();
    }
    return rightMost;
  }

  public String toString() {
    StringBuffer os = new StringBuffer();
    os.append("{");
    if(t1 != null) os.append(t1);
    else os.append("NIL");
    os.append(", ");
    if(t2 != null) os.append(t2);
    else os.append("NIL");
    os.append(", ");
    if(t3 != null) os.append(t3);
    else os.append("NIL");
    os.append(", ");
    if(t4 != null) os.append(t4);
    else os.append("NIL");
    os.append("}");
    return os.toString();
  }
}
