package edu.stanford.nlp.kbp.slotfilling.webqueries;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.TwoDimensionalCounter;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * @author Angel Chang
 */
public class WebSnippetStats {
  TwoDimensionalCounter<String,String> totalSlotWordCounts;
  Map<String,TwoDimensionalCounter<String,String>> slotWordCounts;
  TwoDimensionalCounter<String,String> slotQueryTypeCounts;

  @SuppressWarnings("unused")
  private static final String TOTAL_KEY = "TOTAL";
  private final static int FIELD_SLOT = 0;
  private final static int FIELD_RANK = 1;
  private final static int FIELD_WORD = 2;
  private final static int FIELD_COUNT = 3;
  private final static int FIELD_MAX = 4;

  private static final Pattern delimiterPattern = Pattern.compile("\t");

  public WebSnippetStats()
  {
    slotWordCounts = new HashMap<String,TwoDimensionalCounter<String,String>>();
    totalSlotWordCounts = new TwoDimensionalCounter<String, String>();
    slotQueryTypeCounts = new TwoDimensionalCounter<String, String>();
  }

  public void incrementSlotWordCounts(String queryType, String slot, String word)
  {
    TwoDimensionalCounter<String,String> counter = slotWordCounts.get(queryType);
    if (counter == null) {
      slotWordCounts.put(queryType, counter = new TwoDimensionalCounter<String, String>());
    }
    counter.incrementCount(slot, word);
    totalSlotWordCounts.incrementCount(slot, word);
  }

  public void printSlotWordStats(PrintWriter pw) throws IOException
  {
    printSlotCounts(totalSlotWordCounts, pw);
  }

  public void printSlotWordStats(String queryType, PrintWriter pw) throws IOException
  {
    printSlotCounts(slotWordCounts.get(queryType), pw);
  }

  public void printSlotQueryTypeStats(PrintWriter pw) throws IOException
  {
    printSlotCounts(slotQueryTypeCounts, pw);
  }

  public void printSlotCounts(TwoDimensionalCounter<String, String> slotCounts, PrintWriter pw) throws IOException
  {
    Set<String> slots = slotCounts.firstKeySet();
    for (String slot:slots) {
      Counter<String> counter = slotCounts.getCounter(slot);
      List<String> sortedWords = Counters.toSortedList(counter);
      int i = 0;
      for (String word:sortedWords) {
        i++;
        pw.println(slot + "\t" + i + "\t" + word + "\t" + ((int) counter.getCount(word)));
      }
    }
  }

  public TwoDimensionalCounter<String, String> readSlotStats(String filename) throws IOException
  {
    TwoDimensionalCounter<String,String> counter = new TwoDimensionalCounter<String,String>();
    BufferedReader br = IOUtils.getBufferedFileReader(filename);
    String line;
    int lineno = 0;
    while ((line = br.readLine()) != null) {
      lineno++;
      String[] fields = delimiterPattern.split(line);
      if (fields.length == FIELD_MAX) {
        String slot = fields[FIELD_SLOT];
        String word = fields[FIELD_WORD];
        @SuppressWarnings("unused")
        int rank = Integer.valueOf(fields[FIELD_RANK]);
        int count = Integer.valueOf(fields[FIELD_COUNT]);
        counter.setCount(slot,word,count);
      } else {
        System.err.println("WARNING: Invalid line: Unexpected number of fields "
                + fields.length + ", expected " + FIELD_MAX
                + " (" + filename + ":" + lineno + "): " + line);
      }
    }
    br.close();
    return counter;
  }

  
}
