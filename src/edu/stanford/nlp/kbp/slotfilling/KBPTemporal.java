package edu.stanford.nlp.kbp.slotfilling;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.GregorianCalendar;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.TemporalSentenceExtractor;
import edu.stanford.nlp.kbp.slotfilling.index.KBPField;
import edu.stanford.nlp.kbp.temporal.FindTemporalExpressionsWiki;
import edu.stanford.nlp.kbp.temporal.TemporalWithSpan;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.OpenAddressCounter;
import edu.stanford.nlp.time.TimeExpression;
import edu.stanford.nlp.time.SUTime.IsoDate;
import edu.stanford.nlp.time.SUTime.Range;
import edu.stanford.nlp.time.SUTime.Temporal;
import edu.stanford.nlp.time.SUTime.Time;
import edu.stanford.nlp.topicmodeling.topicflow.graphviz_layout.FileUtils;
import edu.stanford.nlp.util.CollectionUtils;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.StringUtils;

/**
 * Fills KBP temporal slots T1 T2 T3 T4, for more details see the specifications
 * at
 * 
 * @author sonalg
 * 
 */
public class KBPTemporal {

  static Pattern acceptableTokenRegex = Pattern.compile("[\\dX][\\dX][\\dX][\\dX]-?[\\dX]?[\\dX]?-?[\\dX]?[\\dX]?", Pattern.CASE_INSENSITIVE);
  static final int numTokensToCheck = 4;
  static Counter<String> startWords, endWords, googleNgrams;
  // static Logger logger = Logger.getLogger(KBPTemporal.class.toString());
  static boolean precise = false;
  static boolean baseline = false;
  static boolean useStartEnd = false;

  enum STARTOREND {
    START, END, NONE;
  }

  public static TemporalWithSpan generateTemporalfrmDocDate(List<CoreLabel> tokens, String docDate) {

    if (docDate == null)
      return null;
    // past tense or not didn't make any difference
    // for (CoreLabel c : tokens) {
    // if (c.get(CoreAnnotations.PartOfSpeechAnnotation.class).equals("VBD"))
    // return null;
    // }
    Integer year = Integer.parseInt(docDate.substring(0, 4));
    Integer month = Integer.parseInt(docDate.substring(5, 7));
    Integer day = Integer.parseInt(docDate.substring(8, 10));
    return new TemporalWithSpan(null, new IsoDate(year, month, day), new IsoDate(year, month, day), null, null);

  }

  public static void readStartEndFiles(Properties props) throws Exception {
    baseline = PropertiesUtils.getBool(props, Props.TEMPORAL_BASELINE, false);
    useStartEnd = PropertiesUtils.getBool(props, Props.TEMPORAL_USESTARTEND, false);
    Log.setLevel(Level.parse(props.getProperty(Props.TEMPORAL_LOGLEVEL, "SEVERE")));
    precise = Boolean.parseBoolean(props.getProperty("temporal.precise", "false"));
    String startWordsFile = props.getProperty(Props.TEMPORAL_STARTWORDSFILE);
    String endWordsFile = props.getProperty(Props.TEMPORAL_ENDWORDSFILE);
    String ngramFromGoogleFile = props.getProperty(Props.TEMPORAL_GOOGNGRAMSFILE);

    if (startWordsFile != null && endWordsFile != null) {
      startWords = IOUtils.readObjectFromFile(startWordsFile);
      endWords = IOUtils.readObjectFromFile(endWordsFile);
      googleNgrams = IOUtils.readObjectFromFile(ngramFromGoogleFile);

      // Counter<String> sum = Counters.union(startWords, endWords);
      Collection<String> notPresentInGoogleNgrams = CollectionUtils.diff(startWords.keySet(), googleNgrams.keySet());
      notPresentInGoogleNgrams.addAll(CollectionUtils.diff(endWords.keySet(), googleNgrams.keySet()));
      Log.finest("strings not present in googlengrams: " + StringUtils.join(notPresentInGoogleNgrams, "; "));
      Counters.removeKeys(startWords, notPresentInGoogleNgrams);
      Counters.removeKeys(endWords, notPresentInGoogleNgrams);
      Counters.divideInPlace(startWords, googleNgrams);
      Counters.divideInPlace(endWords, googleNgrams);

      // Counters.normalize(startWords);
      // Counters.normalize(endWords);
      Log.severe("number of words in startWords is " + startWords.size());
      Log.severe("number of words in endWords is " + endWords.size());
    } else
      throw new Exception("start or end words file not found");
  }

  public KBPTemporal(Properties props) throws Exception {
    Log.setLevel(Level.parse(props.getProperty(Props.TEMPORAL_LOGLEVEL, "SEVERE")));
  }

  private static double startEndThreshold = 0.1;

  private static STARTOREND getStartOrEnd(List<CoreLabel> tokens, int beginToken, int endToken, String dateVal) {

    List<String> dateNormalizedTokens = FindTemporalExpressionsWiki.normalizeDateTokens(tokens, beginToken, endToken, dateVal);
    Counter<String> sentenceNGrams = new OpenAddressCounter<String>();
    List<Span> startDateSpans = new ArrayList<Span>();
    if (TemporalSentenceExtractor.matchSlotInSentence(dateVal, dateNormalizedTokens, new String[] { dateVal }, startDateSpans))
      Counters.addInPlace(sentenceNGrams, FindTemporalExpressionsWiki.getNGrams(dateNormalizedTokens, startDateSpans));

    // Counters.normalize(sentenceNGrams);

    double startSim = Counters.jaccardCoefficient(startWords, sentenceNGrams);
    double endSim = Counters.jaccardCoefficient(endWords, sentenceNGrams);

    double sum = (startSim + endSim);
    startSim = startSim / sum;
    endSim = endSim / sum;

    if (startSim > endSim + startEndThreshold) {
      Log.info("gets start with " + startSim + " and " + endSim);
      return STARTOREND.START;
    } else if (startSim + startEndThreshold < endSim) {
      Log.info("gets end with " + startSim + " and " + endSim);
      return STARTOREND.END;
    } else {
      Log.info("gets none with " + startSim + " and " + endSim);
      return STARTOREND.NONE;
    }

  }

  private static String getDateVal(Time t) {
    String startDateVal = null;
    if (t != null) {
      startDateVal = t.toISOString();
      if (startDateVal != null && startDateVal.matches(".*\\d.*")) {
        startDateVal = startDateVal.substring(0, Math.min(startDateVal.length(), 10));
        if (!acceptableTokenRegex.matcher(startDateVal).matches())
          startDateVal = null;
      } else
        startDateVal = null;
    }
    return startDateVal;
  }

  private static String fillYear(int year, String date) {
    if (year > 0 && date != null && date.matches("XXXX-[\\dX][\\dX]-[\\dX][\\dX]")) {
      return year + date.substring(4, date.length());
    }
    return date;
  }

  /**
   * if one of the dates in the range has date missing then, add "01" if it is
   * begin date and "30 or 31" if it is end date
   * 
   */
  private static String fillDate(String date, boolean begin) {

    if (date != null && date.matches("[\\dX][\\dX][\\dX][\\dX]-[\\d][\\d]")) {

      if (begin)
        return date + "-01";
      // if it is end
      Calendar calendar = Calendar.getInstance();
      String yearStr = date.substring(0, 4);
      int year;
      if (!yearStr.matches("\\d\\d\\d\\d"))
        // random year!
        year = 2010;
      else
        year = Integer.parseInt(yearStr);
      int month = Integer.parseInt(date.substring(5, 7)) - 1;
      calendar.set(year, month, 1);
      int days = calendar.getActualMaximum(Calendar.DAY_OF_MONTH);
      return date + "-" + days;
    }
    return date;
  }

  private static TemporalWithSpan processTimeExpressions(int tokenBeginNum, int tokenEndNum, List<CoreLabel> tokens, Temporal dateExp, Temporal dateExp2, int tokenBeginNum2,
      int tokenEndNum2) {
    IsoDate t1 = null, t2 = null, t3 = null, t4 = null;

    List<Span> spans = new ArrayList<Span>();
    spans.add(new Span(tokenBeginNum, tokenEndNum));

    if (tokenBeginNum2 >= 0)
      spans.add(new Span(tokenBeginNum2, tokenEndNum2));

    String dateValue = dateExp.toISOString();
    String dateValue2 = dateExp2 != null ? dateExp2.toISOString() : null;

    int year = -1;
    if (dateValue2 != null) {
      if (dateValue2.length() >= 4 && dateValue2.substring(0, 4).matches("\\d\\d\\d\\d") && (dateValue.length() < 4 || !dateValue.substring(0, 4).contains("\\d\\d\\d\\d"))) {
        year = Integer.parseInt(dateValue2.substring(0, 4));

      } else if (dateValue.length() >= 4 && dateValue.substring(0, 4).matches("\\d\\d\\d\\d") && (dateValue2.length() < 4 || !dateValue2.substring(0, 4).contains("\\d\\d\\d\\d"))) {
        year = Integer.parseInt(dateValue2.substring(0, 4));
      }
    }

    Range startEndDates = dateExp.getRange();

    Range startEndDates2 = dateExp2 != null ? dateExp2.getRange() : null;

    if (startEndDates == null && startEndDates2 == null)
      return null;

    String startDateVal = startEndDates != null ? getDateVal(startEndDates.begin()) : null;
    String endDateVal = startEndDates != null ? getDateVal(startEndDates.end()) : null;

    String startDateVal2 = startEndDates2 != null ? getDateVal(startEndDates2.begin()) : null;
    String endDateVal2 = startEndDates2 != null ? getDateVal(startEndDates2.end()) : null;

    dateValue = fillYear(year, dateValue);
    dateValue2 = fillYear(year, dateValue2);

    startDateVal = fillYear(year, startDateVal);
    endDateVal = fillYear(year, endDateVal);
    startDateVal2 = fillYear(year, startDateVal2);
    endDateVal2 = fillYear(year, endDateVal2);

    // when SUTime has a bug, and gives range dates in YYYY-MM format rather
    // than YYYY-MM-DD format. Fill in the date value according to the month
    startDateVal = fillDate(startDateVal, true);
    endDateVal = fillDate(endDateVal, false);
    startDateVal2 = fillDate(startDateVal2, true);
    endDateVal2 = fillDate(endDateVal2, false);

    Log.info("val is " + dateValue + " and start/end dates are " + startDateVal + " and " + endDateVal);
    Log.info("val2 is " + dateValue2 + " and start/end dates are " + startDateVal2 + " and " + endDateVal2);

    if (startDateVal == null && endDateVal == null && startDateVal2 == null && endDateVal2 == null) {
      Log.info("not accepted since both start and end date are null");
      return null;
    }

    Pattern exactDate = Pattern.compile("\\d\\d\\d\\d-\\d\\d-\\d\\d");

    boolean isExactDate = false;
    if (exactDate.matcher(dateValue).matches())
      isExactDate = true;

    boolean isExactDate2 = false;
    if (dateValue2 != null && exactDate.matcher(dateValue2).matches())
      isExactDate2 = true;

    // if there are two dates, just use them such that first date fills T1, T2
    // and second date fills T3, T4
    if (dateExp2 != null) {
      if (isExactDate) {
        t1 = null;
        t2 = new IsoDate(dateValue.substring(0, 4), dateValue.substring(5, 7), dateValue.substring(8, 10));
      } else {
        t1 = new IsoDate(startDateVal.substring(0, 4), startDateVal.substring(5, 7), startDateVal.substring(8, 10));
        t2 = new IsoDate(endDateVal.substring(0, 4), endDateVal.substring(5, 7), endDateVal.substring(8, 10));
      }

      if (isExactDate2) {
        t3 = null;
        t4 = new IsoDate(dateValue2.substring(0, 4), dateValue2.substring(5, 7), dateValue2.substring(8, 10));
      } else {
        t3 = new IsoDate(startDateVal2.substring(0, 4), startDateVal2.substring(5, 7), startDateVal2.substring(8, 10));
        t4 = new IsoDate(endDateVal2.substring(0, 4), endDateVal2.substring(5, 7), endDateVal2.substring(8, 10));
      }

      return new TemporalWithSpan(t1, t2, t3, t4, spans);
    }

    if (useStartEnd) {
      STARTOREND sore = getStartOrEnd(tokens, tokenBeginNum, tokenEndNum, dateValue);
      if (sore.equals(STARTOREND.START)) {
        t1 = new IsoDate(startDateVal.substring(0, 4), startDateVal.substring(5, 7), startDateVal.substring(8, 10));
        t2 = new IsoDate(endDateVal.substring(0, 4), endDateVal.substring(5, 7), endDateVal.substring(8, 10));
        t3 = null;
        t4 = null;
      } else if (sore.equals(STARTOREND.END)) {
        t1 = null;
        t2 = null;
        t3 = new IsoDate(startDateVal.substring(0, 4), startDateVal.substring(5, 7), startDateVal.substring(8, 10));
        t4 = new IsoDate(endDateVal.substring(0, 4), endDateVal.substring(5, 7), endDateVal.substring(8, 10));
      } else {
        t1 = null;
        // this might seem like a bug, but it is not
        t2 = new IsoDate(endDateVal.substring(0, 4), endDateVal.substring(5, 7), endDateVal.substring(8, 10));
        t3 = new IsoDate(startDateVal.substring(0, 4), startDateVal.substring(5, 7), startDateVal.substring(8, 10));
        t4 = null;
      }

    } else {
      t1 = null;
      // this might seem like a bug, but it is not
      t2 = new IsoDate(endDateVal.substring(0, 4), endDateVal.substring(5, 7), endDateVal.substring(8, 10));
      t3 = new IsoDate(startDateVal.substring(0, 4), startDateVal.substring(5, 7), startDateVal.substring(8, 10));
      t4 = null;
    }

    // Pattern exactPattern = Pattern.compile("on|in",
    // Pattern.CASE_INSENSITIVE);

    /*if (tokenBeginNum > 0 && exactPattern.matcher(tokens.get(tokenBeginNum - 1).word()).matches()) {
            if (isExactDate) {
              t1 = null;
              t2 = new IsoDate(dateValue.substring(0, 4), dateValue.substring(5, 7), dateValue.substring(8, 10));
              t3 = new IsoDate(dateValue.substring(0, 4), dateValue.substring(5, 7), dateValue.substring(8, 10));
              t4 = null;

            } else
      if (sore.equals(STARTOREND.START)) {
        t1 = new IsoDate(startDateVal.substring(0, 4), startDateVal.substring(5, 7), startDateVal.substring(8, 10));
        t2 = new IsoDate(endDateVal.substring(0, 4), endDateVal.substring(5, 7), endDateVal.substring(8, 10));
        t3 = null;
        t4 = null;
      } else if (sore.equals(STARTOREND.END)) {
        t1 = null;
        t2 = null;
        t3 = new IsoDate(startDateVal.substring(0, 4), startDateVal.substring(5, 7), startDateVal.substring(8, 10));
        t4 = new IsoDate(endDateVal.substring(0, 4), endDateVal.substring(5, 7), endDateVal.substring(8, 10));
      } else {
        t1 = new IsoDate(startDateVal.substring(0, 4), startDateVal.substring(5, 7), startDateVal.substring(8, 10));
        t2 = new IsoDate(endDateVal.substring(0, 4), endDateVal.substring(5, 7), endDateVal.substring(8, 10));
        t3 = new IsoDate(startDateVal.substring(0, 4), startDateVal.substring(5, 7), startDateVal.substring(8, 10));
        t4 = new IsoDate(endDateVal.substring(0, 4), endDateVal.substring(5, 7), endDateVal.substring(8, 10));
      }
      Log.fine("Matched ON!! original date value is " + dateValue + " and start and end dates are " + startDateVal + "\t" + endDateVal);
      // "has been .* for " ago

      return new TemporalWithSpan(t1, t2, t3, t4, spans);
    }*/
    // Pattern starting = Pattern.compile("from|since",
    // Pattern.CASE_INSENSITIVE);
    // Pattern end = Pattern.compile("till|until", Pattern.CASE_INSENSITIVE);
    //
    // if (sore.equals(STARTOREND.START) || tokenBeginNum > 0 &&
    // starting.matcher(tokens.get(tokenBeginNum - 1).word()).matches()) {
    // if (isExactDate) {
    // t1 = null;
    // t2 = new IsoDate(dateValue.substring(0, 4), dateValue.substring(5, 7),
    // dateValue.substring(8, 10));
    // t3 = null;
    // t4 = null;
    // } else {
    // t1 = startDateVal != null ? new IsoDate(startDateVal.substring(0, 4),
    // startDateVal.substring(5, 7), startDateVal.substring(8, 10)) : null;
    // t2 = endDateVal != null ? new IsoDate(endDateVal.substring(0, 4),
    // endDateVal.substring(5, 7), endDateVal.substring(8, 10)) : null;
    // t3 = null;
    // t4 = null;
    // }
    // } else if (sore.equals(STARTOREND.END) || (tokenBeginNum > 0 &&
    // end.matcher(tokens.get(tokenBeginNum - 1).word()).matches())) {
    // if (isExactDate) {
    // t1 = null;
    // t2 = null;
    // t3 = new IsoDate(dateValue.substring(0, 4), dateValue.substring(5, 7),
    // dateValue.substring(8, 10));
    // t4 = null;
    // } else {
    // t1 = null;
    // t2 = null;
    // t3 = startDateVal != null ? new IsoDate(startDateVal.substring(0, 4),
    // startDateVal.substring(5, 7), startDateVal.substring(8, 10)) : null;
    // t4 = endDateVal != null ? new IsoDate(endDateVal.substring(0, 4),
    // endDateVal.substring(5, 7), endDateVal.substring(8, 10)) : null;
    // }
    // } else {
    // // if it is none
    // if (isExactDate) {
    // t1 = null;
    // t2 = new IsoDate(dateValue.substring(0, 4), dateValue.substring(5, 7),
    // dateValue.substring(8, 10));
    // t3 = new IsoDate(dateValue.substring(0, 4), dateValue.substring(5, 7),
    // dateValue.substring(8, 10));
    // t4 = null;
    // } else {
    // if (!precise) {
    // t1 = startDateVal != null ? new IsoDate(startDateVal.substring(0, 4),
    // startDateVal.substring(5, 7), startDateVal.substring(8, 10)) : null;
    // t2 = endDateVal != null ? new IsoDate(endDateVal.substring(0, 4),
    // endDateVal.substring(5, 7), endDateVal.substring(8, 10)) : null;
    // t3 = startDateVal != null ? new IsoDate(startDateVal.substring(0, 4),
    // startDateVal.substring(5, 7), startDateVal.substring(8, 10)) : null;
    // t4 = endDateVal != null ? new IsoDate(endDateVal.substring(0, 4),
    // endDateVal.substring(5, 7), endDateVal.substring(8, 10)) : null;
    // }
    // }
    // }

    // if all four null, then don't create the object
    if (t1 == null && t2 == null && t3 == null && t4 == null)
      return null;

    return new TemporalWithSpan(t1, t2, t3, t4, spans);
  }

  public static List<TemporalWithSpan> getSlotValues(List<CoreMap> timeCluster, List<CoreLabel> tokens) throws Exception {

    if (timeCluster.size() > 2)
      throw new Exception("can't process cluster whose size is greater than 2. How did it reach here?");

    List<TemporalWithSpan> spansInCluster = new ArrayList<TemporalWithSpan>();
    if (baseline)
      return spansInCluster;
    if (timeCluster.size() == 2)
      System.out.println("\n\n time cluster size is equal to 2 ");

    CoreMap timeExp = timeCluster.get(0);
    // assuming max size of time cluster is 2
    CoreMap timeExp2 = timeCluster.size() > 1 ? timeCluster.get(1) : null;

    TimeExpression tm = timeExp.get(TimeExpression.Annotation.class);

    Temporal t = tm.getTemporal();
    Temporal t2 = timeExp2 != null ? timeExp2.get(TimeExpression.Annotation.class).getTemporal() : null;
    int tokenBeginNum = timeExp.get(CoreAnnotations.TokenBeginAnnotation.class);
    int tokenEndNum = timeExp.get(CoreAnnotations.TokenEndAnnotation.class);

    int tokenBeginNum2 = timeExp2 != null ? timeExp2.get(CoreAnnotations.TokenBeginAnnotation.class) : -1;
    int tokenEndNum2 = timeExp2 != null ? timeExp2.get(CoreAnnotations.TokenEndAnnotation.class) : -1;

    TemporalWithSpan s = processTimeExpressions(tokenBeginNum, tokenEndNum, tokens, t, t2, tokenBeginNum2, tokenEndNum2);
    Log.info("The temporal slot values are: " + s != null ? s.toString() : null);
    if (s != null)
      spansInCluster.add(s);

    Logger Log = Logger.getLogger(KBPTemporal.class.getName());
    Log.setLevel(Level.SEVERE);
    return spansInCluster;
  }

  public static void normalize(String date) {
    // assuming northern hemisphere - fall is from Sept to Nov
    if (Pattern.matches("\\d\\d\\d\\d-FA", date) || date.equals("XXXX-FA")) {
      String year = date.substring(0, 4);
      @SuppressWarnings("unused")
      String startDate = year + "-09-01";
      @SuppressWarnings("unused")
      String endDate = year + "-11-30";
    }
    // assuming northern hemisphere - spring is March till May
    else if (Pattern.matches("\\d\\d\\d\\d-SP", date) || date.equals("XXXX-SP")) {
      String year = date.substring(0, 4);
      @SuppressWarnings("unused")
      String startDate = year + "-03-01";
      @SuppressWarnings("unused")
      String endDate = year + "-05-31";

    } else if (Pattern.matches("\\d\\d\\d\\d-SU", date) || date.equals("XXXX-SU")) {
      String year = date.substring(0, 4);
      @SuppressWarnings("unused")
      String startDate = year + "-06-01";
      @SuppressWarnings("unused")
      String endDate = year + "-08-31";
    } else if (Pattern.matches("\\d\\d\\d\\d-WI", date) || date.equals("XXXX-WI")) {
      String year = date.substring(0, 4);
      @SuppressWarnings("unused")
      String startDate;
      if (StringUtils.isNumeric(year))
        startDate = (Integer.parseInt(year) - 1) + "-12-01";
      else
        startDate = year + "-12-01";
      @SuppressWarnings("unused")
      String endDate = year + "-02-31";
    }

  }

  public static Pair<String, String> normalizeDate(String date) {

    Pair<String, String> startEndDate = new Pair<String, String>();
    if (Pattern.matches("\\d\\d\\d\\d-\\d\\d-\\d\\d", date)) {
      startEndDate.setFirst(date);
      startEndDate.setSecond(date);
    } else if (Pattern.matches("\\d\\d\\d\\d-\\d\\d", date)) {
      Calendar cal = new GregorianCalendar();
      int maxDayOfMonth = cal.getActualMaximum(Calendar.DAY_OF_MONTH);
      startEndDate.setFirst(date + "-01");
      startEndDate.setSecond(date + "-" + maxDayOfMonth);
    } else if (Pattern.matches("\\d\\d\\d\\d", date)) {
      startEndDate.setFirst(date + "-01-01");
      startEndDate.setSecond(date + "-12-31");
    }

    return startEndDate;
  }

  public static void getDocumentDates(String luceneIndexPath, String outputFile) {
    try {
      BufferedWriter w = new BufferedWriter(new FileWriter(outputFile));
      Directory dir = FSDirectory.open(new File(luceneIndexPath));
      IndexReader r = IndexReader.open(dir);
      for (int i = 0; i < r.numDocs(); i++) {
        w.write(r.document(i).get(KBPField.DOCID.fieldName()) + "\t" + r.document(i).get("datetime") + "\n");
      }
      w.close();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  /*
   * this method is a hack. It reads a file that has document id and document
   * date-time, which is same as in the lucence index. It was created so as to
   * not run the pipeline of annotation again. Initially document datetime was
   * not stored in the annotation cache.
   */
  public static HashMap<String, String> readDocumentDates() {
    try {
      HashMap<String, String> documentDates = new HashMap<String, String>();
      for (String line : FileUtils.readLines(new File("/home/sonalg/javanlp/docDates.txt"))) {
        String[] tokens = line.trim().split("\\s+");
        String docId = tokens[0].trim();
        String datetime = tokens[1].trim();
        documentDates.put(docId, datetime);
      }
      return documentDates;
    } catch (Exception e) {
      e.printStackTrace();
    }
    return null;
  }

  public static void main(String[] args) {
    String luceneIndexPath = "/u/nlp/data/TAC-KBP2010/indices/TAC_2010_KBP_Source_Data_Index_Cached";
    KBPTemporal.getDocumentDates(luceneIndexPath, "docDates.txt");
  }

}
