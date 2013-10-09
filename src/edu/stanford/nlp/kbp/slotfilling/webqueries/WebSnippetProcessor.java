package edu.stanford.nlp.kbp.slotfilling.webqueries;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.slotfilling.common.AntecedentGenerator;
import edu.stanford.nlp.kbp.slotfilling.common.EntityType;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations;
import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.KBPSlot;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Utils;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.KBPDomReader;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.TaskXMLParser;
import edu.stanford.nlp.kbp.slotfilling.index.KBPAnnotationSerializer;
import edu.stanford.nlp.kbp.slotfilling.index.SentenceCacher;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.tokensregex.MultiWordStringMatcher;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationSerializer;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.wikipedia.pipeline.TokenizerPostProcessorAnnotator;
import org.apache.commons.lang.StringEscapeUtils;
import org.htmlparser.util.Translate;
import org.xml.sax.SAXException;

import java.io.*;
import java.text.Normalizer;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.xml.parsers.ParserConfigurationException;

/**
 * Process web snippets
 *
 * @author Angel Chang
 * @author Mihai
 */
public class WebSnippetProcessor {
  private static final Pattern delimiterPattern = Pattern.compile("\t");
  private static Map<String, String> cmdMap =
          new HashMap<String, String>();

  static {
    cmdMap.put("printStats", "print statistics for web snippets");
    cmdMap.put("cleanSnippets", "cleans web snippets");
    cmdMap.put("printSnippets", "print web snippets (mostly for debugging)");
    cmdMap.put("saveAnnotations", "save web snippet annotation");
    cmdMap.put("printAnnotations", "read and print web snippet annotation");
    cmdMap.put("printSamples", "read samples for classifier");
    cmdMap.put("toCacheTest", "converts web snippets for NEUtral queries to annotations in the KBPReader cache format (include annotation!)");
    cmdMap.put("toCacheTrain", "converts web snippets for POSitive queries to annotations in the KBPReader cache format (include annotation!)");
  }

  protected boolean autoCleanSnippets = true;
  protected int formatVersion = 1;

  public WebSnippetsMap readSnippets(String filename) throws IOException {
    WebSnippetReader snippetReader = new WebSnippetReader();
    snippetReader.init(null);
    processSnippets(filename, snippetReader);
    snippetReader.finish();
    return snippetReader.snippetsMap;
  }

  public void cleanSnippets(String filename, Pattern filterPattern, Properties props) throws IOException {
    WebSnippetWriter snippetWriter = new WebSnippetWriter(System.out);
    WebSnippetCleaner snippetCleaner = new WebSnippetCleaner(snippetWriter);
    WebSnippetFilterer snippetFilterer = new WebSnippetFilterer(snippetCleaner);
    snippetWriter.init(props);
    snippetCleaner.init(props);
    snippetFilterer.init(props);
    processSnippets(filename, filterPattern, snippetFilterer);
    snippetFilterer.finish();
    snippetCleaner.finish();
    snippetWriter.finish();
  }

  public void printSamples(String filename, Pattern filterPattern, Properties props) throws IOException {
    WebSnippetSampleConverter sampleConverter = new WebSnippetSampleConverter();
    WebSnippetCleaner snippetCleaner = new WebSnippetCleaner(sampleConverter);
    autoCleanSnippets = false;
    snippetCleaner.discardPunctuation = true;
    snippetCleaner.lowerCase = true;
    sampleConverter.init(props);
    snippetCleaner.init(props);
    processSnippets(filename, filterPattern, snippetCleaner);
    snippetCleaner.finish();
    sampleConverter.finish();
  }

  public void printSnippets(String filename, Pattern filterPattern, Properties props) throws IOException {
    WebSnippetWriter snippetWriter = new WebSnippetWriter(System.out);
    WebSnippetFilterer snippetFilterer = new WebSnippetFilterer(snippetWriter);
    snippetWriter.init(props);
    snippetFilterer.init(props);
    processSnippets(filename, filterPattern, snippetFilterer);
    snippetFilterer.finish();
    snippetWriter.finish();
  }
  
  public void snippetsToCacheTest(String inputFile, String cacheDir, Map<String, EntityType> entityTypes, Properties props) throws IOException {
    WebSnippetCacher snippetCacher = new WebSnippetCacher(cacheDir, entityTypes, true, props);
    processSnippets(inputFile, (Pattern) null, snippetCacher);
    snippetCacher.finish();
  }
  
  public void snippetsToCacheTrain(String inputFile, String cacheDir, Map<String, EntityType> entityTypes, Properties props) throws IOException {
    WebSnippetCacher snippetCacher = new WebSnippetCacher(cacheDir, entityTypes, false, props);
    processSnippets(inputFile, (Pattern) null, snippetCacher);
    snippetCacher.finish();
    System.err.println("Found " + snippetCacher.docCount + " valid documents. Skipped " + snippetCacher.queriesSkipped + "/" + snippetCacher.queriesValid + " queries.");
  }

  public void printStats(String filename, Pattern filterPattern, Properties props) throws IOException {
    WebSnippetWordCounter wordCounter = new WebSnippetWordCounter();
    WebSnippetCleaner snippetCleaner = new WebSnippetCleaner(wordCounter);
    WebSnippetFilterer snippetFilterer = new WebSnippetFilterer(snippetCleaner);
    autoCleanSnippets = false;
    snippetCleaner.keepBold = true;
    snippetCleaner.keepEm = true;
    snippetCleaner.keepEllipsis = false;
    snippetCleaner.normalize = true;
    snippetCleaner.unescapeHtml = true;
    snippetCleaner.discardPunctuation = true;
    snippetCleaner.lowerCase = true;
    wordCounter.init(props);
    snippetCleaner.init(props);
    snippetFilterer.init(props);
    processSnippets(filename, filterPattern, snippetFilterer);
    snippetFilterer.finish();
    snippetCleaner.finish();
    wordCounter.finish();
  }

  // Takes input file of snippets and annotates it, storing it to the outputFile
  public void annotateSnippets(StanfordCoreNLP pipeline, String inputFile, String outputFile, String partialCachedFile) throws IOException {
    WebSnippetAnnotator webSnippetAnnotator = new WebSnippetAnnotator(pipeline, outputFile, partialCachedFile);
    webSnippetAnnotator.init(null);
    processSnippets(inputFile, webSnippetAnnotator);
    webSnippetAnnotator.finish();
  }

  /* Prints annotated snippets */
  public void printAnnotatedSnippets(String inputFile) throws IOException, ClassNotFoundException {
    PrintStream out = System.out;
    ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(IOUtils.getFileInputStream(inputFile)));
    Object obj;
    int i = 0;
    while ((obj = ois.readObject()) != null) {
      i++;
      /*WebSnippets.RelationMentionSnippets snippets = ErasureUtils.<WebSnippets.RelationMentionSnippets> uncheckedCast(obj);
      out.println("Read snippets group #" + i + " with " + snippets.snippets.size() + " snippets ");
      out.println("   for " + snippets.getHeader()); */
      String header = ErasureUtils.<String> uncheckedCast(obj);
      out.println("Read snippets group #" + i);
      out.println("   for " + header);
      obj = ois.readObject();
      List<Annotation> annotations = ErasureUtils.<List<Annotation>> uncheckedCast(obj);
      out.println("   got " + annotations.size() + " annotations ");
      int j=0;
      for (Annotation annotation:annotations) {
        j++;
        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        out.println("Annotation #" + i +"-" + j + " has " + sentences.size() + " sentences ");
        List<IntPair> marked = annotation.get(KBPAnnotations.MarkedPositionsAnnotation.class);
        if (marked != null) {
          out.println(" and " + marked.size() + " marked mentions");
        }
        for (CoreMap sentence:sentences) {
          Utils.printSentence(out, sentence);
        }
      }
    }
    ois.close();
  }
/*
  public static void annotateSnippets(BaselineNLProcessor pipeline, String dir, Pattern filterPattern, String outputDir) throws IOException
  {
    File file = new File(dir);
    if (file.isDirectory()) {
      String[] filenames = file.list();
      for (String f:filenames) {
        annotateSnippets(pipeline, f, filterPattern, outputDir + "/" + f);
      }
    } else {
      String filename = dir;
      if (filterPattern == null || filterPattern.matcher(filename).matches()) {
        annotateSnippets(pipeline, filename, outputDir);
      }
    }
  }
*/
  public void processSnippets(String filename, String filterPatternStr, WebSnippetHandler handler) throws IOException
  {
    Pattern filterPattern = (filterPatternStr != null)? Pattern.compile(filterPatternStr):null;
    processSnippets(filename, filterPattern, handler);
  }

  public void processSnippets(String filename, Pattern filterPattern, WebSnippetHandler handler) throws IOException
  {
    File file = new File(filename);
    if (file.isDirectory()) {
      File[] files = file.listFiles();
      for (File f:files) {
        processSnippets(f.getAbsolutePath(), filterPattern, handler);
      }
    } else {
      if (filterPattern == null || filterPattern.matcher(filename).matches()) {
        processSnippets(filename, handler);
      }
    }
  }

  public void processSnippets(String filename, WebSnippetHandler handler) throws IOException
  {
    if (formatVersion == 0) {
      processSnippetsV0(filename, handler);
    } else {
      processSnippetsV1(filename, handler);
    }
  }

  /* The entries in the file are separated by blank lines and headed by
     tab-delimited slot name, first entity, second entity, keyword (can
     be empty) and query type, one of
     - EE -- entity-to-entity queries, as before (empty keyword)
     - POS -- positive queries (entity one, two and keyword)
     - NEU -- neutral queries (entity one and keyword)
     - NEG -- negated queries (entity one, minus two and keyword)
     - NGA -- strongly (all) negated (entity one, minus each term in two, and keyword)

     The next line is tab-separated zero, estimated number of search
     results (this is totally nuts -- sometimes negative, sometimes
     higher than the actual number of results, but that's what the
     internal framework returns, sadly -- I checked), followed
     by the actual query that was run, for reference.

    Next are up to ten lines, containing tab-separated rank, url and snippet.
    */
  public void processSnippetsV1(String filename, WebSnippetHandler handler) throws IOException
  {
    BufferedReader br = IOUtils.getBufferedFileReader(filename);
    String line;
    String relation = null;
    RelationMentionSnippets snippets = null;
    int lineno = 0;
    while ((line = br.readLine()) != null) {
      lineno++;
      line = line.trim();
      if (line.length() == 0) {
        // Empty line, expecting new snippet
        if (snippets != null) { handler.finishMention(snippets); }
        snippets = null;
      } else if (snippets == null) {
        String[] fields = delimiterPattern.split(line);
        if (fields.length == 5) {
          // Web snippets with negative/postive examples:
          // slot entity querytype(pos/neg/neu) filler keyword info(number of results, time)
          String slot = fields[0];
          String entity = fields[1];
          String filler = fields[2];
          String keyword = fields[3];
          String queryTypeName = fields[4];
          if (relation == null || !relation.equals(slot)) {
            if (relation != null) { handler.finishRelation(relation); }
            relation = slot;
            handler.startRelation(relation);
          }
          snippets = new RelationMentionSnippets(queryTypeName, entity, slot, filler);
          snippets.keyword = keyword;
          handler.startMention(snippets);
        } else {
          System.err.println("WARNING: Invalid line: Unexpected number of fields "
                + fields.length + ", expected " + 5 + " for query description"
                + " (" + filename + ":" + lineno + "): " + line);
        }
      } else {
        String[] fields = delimiterPattern.split(line);
        if (fields.length == 3) {
          try {
            int rank = Integer.parseInt(fields[0]);
            if (rank == 0) {
              // Info about queries
              snippets.totalResultsCount = Long.parseLong(fields[1]);
              snippets.queryString = fields[2];              
            } else {
              String link = fields[1];
              String text = fields[2];
              // lets remove markup from these snippets
              if (autoCleanSnippets) {
                text = WebSnippetCleaner.cleanMarkup(text, true);
              }
              handler.processSnippet(snippets, new WebSnippet(rank, link, text));
            }
          } catch (IllegalArgumentException ex) {
            System.err.println("WARNING: Invalid line - error extracting snippet info "
                  + " (" + filename + ":" + lineno + "): " + line);
            ex.printStackTrace(System.err);
          }
        } else {
          System.err.println("WARNING: Invalid line: Unexpected number of fields "
                + fields.length + ", expected " + 5 + " for snippet"
                + " (" + filename + ":" + lineno + "): " + line);
        }
      }
    }
    if (snippets != null) { handler.finishMention(snippets); }
    if (relation != null) { handler.finishRelation(relation); }
    br.close();
  }

  /**
   * Processes old style snippets (format v0)
   * @param filename
   * @param handler
   * @throws IOException
   */
  public void processSnippetsV0(String filename, WebSnippetHandler handler) throws IOException
  {
    BufferedReader br = IOUtils.getBufferedFileReader(filename);
    String line;
    String relation = null;
    RelationMentionSnippets snippets = null;
    int lineno = 0;
    while ((line = br.readLine()) != null) {
      lineno++;
      line = line.trim();
      if (line.length() == 0) {
        // Empty line, expecting new snippet
        if (snippets != null) { handler.finishMention(snippets); }
        snippets = null;
      } else if (snippets == null) {
        String[] fields = delimiterPattern.split(line);
        if (fields.length == 3) {
          // Web snippets with basic slot information: slot entity filler
          String slot = fields[0];
          String entity = fields[1];
          String filler = fields[2];
          if (relation == null || !relation.equals(slot)) {
            if (relation != null) { handler.finishRelation(relation); }
            relation = slot;
            handler.startRelation(relation);
          }
          snippets = new RelationMentionSnippets(RelationMentionSnippets.QueryType.EE, entity, slot, filler);
          handler.startMention(snippets);
        } else if (fields.length == 6) {
          // Web snippets with negative/postive examples:
          // slot entity querytype(pos/neg/neu) filler keyword info(number of results, time)
          String slot = fields[0];
          String entity = fields[1];
          String queryTypeName = fields[2];
          String filler = fields[3];
          String keyword = fields[4];
          String info = fields[5];
          if (relation == null || !relation.equals(slot)) {
            if (relation != null) { handler.finishRelation(relation); }
            relation = slot;
            handler.startRelation(relation);
          }
          snippets = new RelationMentionSnippets(queryTypeName, entity, slot, filler);
          snippets.keyword = keyword;
          snippets.resultsInfo = info;
          handler.startMention(snippets);
        } else {
          System.err.println("WARNING: Invalid line: Unexpected number of fields "
                + fields.length + ", expected " + 3
                + " (" + filename + ":" + lineno + "): " + line);
        }
      } else {
        handler.processSnippet(snippets, new WebSnippet(line));
      }
    }
    if (snippets != null) { handler.finishMention(snippets); }
    if (relation != null) { handler.finishRelation(relation); }
    br.close();
  }

  public static interface WebSnippetHandler
  {
    public void init(Properties props);
    public void startRelation(String relation);
    public void finishRelation(String relation);
    public void startMention(RelationMentionSnippets snippets);
    public void finishMention(RelationMentionSnippets snippets);
    public void processSnippet(RelationMentionSnippets snippets, WebSnippet snippet);
    public void finish();
  }

  public static abstract class AbstractWebSnippetHandler implements WebSnippetHandler
  {
    public void init(Properties props) {}
    public void startRelation(String relation) {}
    public void finishRelation(String relation) {}
    public void startMention(RelationMentionSnippets snippets) {}
    public void finishMention(RelationMentionSnippets snippets) {}
    public void finish() {}
  }

  public static abstract class ChainedWebSnippetHandler extends AbstractWebSnippetHandler
  {
    WebSnippetHandler nextHandler;
    public ChainedWebSnippetHandler(WebSnippetHandler nextHandler) {
      this.nextHandler = nextHandler;
    }

    public void startMention(RelationMentionSnippets snippets)
    {
      if (nextHandler != null) {
        nextHandler.startMention(snippets);
      }
    }

    public void finishMention(RelationMentionSnippets snippets)
    {
      if (nextHandler != null) {
        nextHandler.finishMention(snippets);
      }
    }

    public void startRelation(String relation)
    {
      if (nextHandler != null) {
        nextHandler.startRelation(relation);
      }
    }

    public void finishRelation(String relation)
    {
      if (nextHandler != null) {
        nextHandler.startRelation(relation);
      }
    }

  }

  public static class WebSnippetReader extends AbstractWebSnippetHandler
  {
    protected WebSnippetsMap snippetsMap;

    public WebSnippetReader() {
      snippetsMap = new WebSnippetsMap();
    }

    public void processSnippet(RelationMentionSnippets snippets, WebSnippet snippet)
    {
      snippets.add(snippet);
    }

    public void startMention(RelationMentionSnippets snippets)
    {
      snippetsMap.addSnippets(snippets);
    }
  }

  public static class WebSnippetWordCounter extends AbstractWebSnippetHandler
  {
    WebSnippetMasker textMasker;
    String wordStatsFile;
    PrintWriter wordStatsWriter;
    String queryTypeStatsFile;
    PrintWriter queryTypeStatsWriter;
    WebSnippetStats stats;
    private static final Pattern whitespacePattern = Pattern.compile("\\s+");

    public WebSnippetWordCounter() {
      stats = new WebSnippetStats();
    }

    public WebSnippetWordCounter(PrintWriter wordStatsWriter, PrintWriter queryTypeStatsWriter) {
      stats = new WebSnippetStats();
      this.wordStatsWriter = wordStatsWriter;
      this.queryTypeStatsWriter = queryTypeStatsWriter;
    }

    @Override
    public void init(Properties props) {
      super.init(props);
      textMasker = new WebSnippetMasker(props);
      try {
        wordStatsFile = props.getProperty("kbp.websnippets.stats.wordStats");
        wordStatsWriter = (wordStatsFile != null)? IOUtils.getPrintWriter(wordStatsFile): new PrintWriter(System.out);
        queryTypeStatsFile = props.getProperty("kbp.websnippets.stats.queryTypeStats");
        queryTypeStatsWriter = (queryTypeStatsFile != null)? IOUtils.getPrintWriter(queryTypeStatsFile): new PrintWriter(System.out);
      } catch (IOException ex) {
        throw new RuntimeException(ex);
      }
    }

    public void processSnippet(RelationMentionSnippets snippets, WebSnippet snippet)
    {
      String text = snippet.getText();
      if (textMasker != null) {
        text = textMasker.getMaskedText(text, snippets.getEntityName(), snippets.getSlotValue());
      }
      String[] tokens = whitespacePattern.split(text);
      for (String token:tokens) {
        stats.incrementSlotWordCounts(snippets.getQueryTypeName(), snippets.slotName, token);
      }
      stats.slotQueryTypeCounts.incrementCount(snippets.slotName, snippets.getQueryTypeName());
    }

    public void finish()
    {
      if (wordStatsWriter != null) {
        try {
          // TODO: Print word stats for other types of QueryTypes
          //stats.printSlotWordStats(RelationMentionSnippets.QueryType.EE.name(), wordStatsWriter);
          stats.printSlotWordStats(wordStatsWriter);
        } catch (IOException ex) {
          throw new RuntimeException(ex);
        }
        if (wordStatsFile != null) {
          wordStatsWriter.close();
        }
      }
      if (queryTypeStatsWriter != null) {
        try {
          stats.printSlotQueryTypeStats(queryTypeStatsWriter);
        } catch (IOException ex) {
          throw new RuntimeException(ex);
        }
        if (queryTypeStatsFile != null) {
          queryTypeStatsWriter.close();
        }
      }
    }
  }

  /**
   * Cleans web snippet of the following
   * - HTML tags (except maybe <em></em> marks)
   * - Stuff like [ Translate this page ]
   */
  public static class WebSnippetCleaner extends ChainedWebSnippetHandler
  {
    boolean discardPunctuation = false;
    boolean normalize = true;
    boolean keepEm = true;
    boolean unescapeHtml = true;
    boolean keepBold = true;
    boolean keepEllipsis = true;
    boolean lowerCase = false;

    private static final Pattern whitespacePattern = Pattern.compile("\\s+");
    private static final Pattern punctPattern = Pattern.compile("[^A-Za-z0-9]+");
//    private static final Pattern punctPattern = Pattern.compile("\\p{Punct}+");
    private static final Pattern translatePattern = Pattern.compile("\\s*\\[\\s*Translate\\s*this\\s*page\\s*\\]", Pattern.CASE_INSENSITIVE);
    private static final Pattern markupPattern = Pattern.compile("<.*?>");
    private static final Pattern markupNoEmPattern = Pattern.compile("(?!<em>|</em>)<.*?>");
    private static final Pattern ellipsisPattern = Pattern.compile("<b>\\.\\.\\.+</b>");
    private static final Pattern markupNoBPattern = Pattern.compile("(?!<b>|</b>)<.*?>");
    private static final Pattern markupNoBEmPattern = Pattern.compile("(?!<b>|</b>|<em>|</em>)<.*?>");
    private static final Pattern markupEllipsisPattern = Pattern.compile("<.*?>.*?</.*?>|\\.\\.\\.");
    private static final Pattern markupPairPattern = Pattern.compile("<.*?>.*?</.*?>");

    public WebSnippetCleaner(WebSnippetHandler nextHandler) {
      super(nextHandler);
    }

    @Override
    public void init(Properties props) {
      super.init(props);
      discardPunctuation = Boolean.parseBoolean(props.getProperty("kbp.websnippets.clean.discardPunctuation", Boolean.toString(discardPunctuation)));
      normalize = Boolean.parseBoolean(props.getProperty("kbp.websnippets.clean.normalize", Boolean.toString(normalize)));
      keepEm = Boolean.parseBoolean(props.getProperty("kbp.websnippets.clean.keepEm", Boolean.toString(keepEm)));
      unescapeHtml = Boolean.parseBoolean(props.getProperty("kbp.websnippets.clean.unescapeHtml", Boolean.toString(unescapeHtml)));
      keepBold = Boolean.parseBoolean(props.getProperty("kbp.websnippets.clean.keepBold", Boolean.toString(keepBold)));
      keepEllipsis = Boolean.parseBoolean(props.getProperty("kbp.websnippets.clean.keepEllipsis", Boolean.toString(keepEllipsis)));
      lowerCase = Boolean.parseBoolean(props.getProperty("kbp.websnippets.clean.lowerCase", Boolean.toString(lowerCase)));
    }

    public static String cleanMarkup(String text, boolean unescapeHtml) {
      text = markupPattern.matcher(text).replaceAll(" ");
      text = whitespacePattern.matcher(text).replaceAll(" ");
      if (unescapeHtml) {
        text = StringEscapeUtils.unescapeHtml(text);
      }
      return text;
    }

    public String cleanSnippet(WebSnippet snippet) {
      // Discard markup and punctuation
      String snippetText = snippet.getText();
      return cleanSnippet(snippetText);
    }

    public String cleanSnippet(String snippetText) {
      if (unescapeHtml) {
        snippetText = StringEscapeUtils.unescapeHtml(snippetText);
      }
      // Discard markup and punctuation
      if (normalize) {
        if (discardPunctuation) {
          snippetText = Normalizer.normalize(snippetText, Normalizer.Form.NFKD);
        } else {
          snippetText = Normalizer.normalize(snippetText, Normalizer.Form.NFKC);
        }
      }
      snippetText = ellipsisPattern.matcher(snippetText).replaceAll("...");
      if (keepEm) {
        if (keepBold) {
          snippetText = markupNoBEmPattern.matcher(snippetText).replaceAll(" ");
        } else {
          snippetText = markupNoEmPattern.matcher(snippetText).replaceAll(" ");
        }
      } else {
        if (keepBold) {
          snippetText = markupNoBPattern.matcher(snippetText).replaceAll(" ");
        } else {
          snippetText = markupPattern.matcher(snippetText).replaceAll(" ");
        }
      }
      snippetText = Translate.decode(snippetText);
      if (discardPunctuation) {
        Pattern skipPattern = (keepEllipsis)? markupEllipsisPattern:markupPairPattern;
        Pattern discardPattern = punctPattern;
        Matcher matcher = skipPattern.matcher(snippetText);
        StringBuilder sb = new StringBuilder();
        int segStart = 0;
        while (matcher.find()) {
          int matchStart = matcher.start();
          int matchEnd = matcher.end();
          if (matchStart > segStart) {
            String seg = snippetText.substring(segStart, matchStart);
            sb.append(discardPattern.matcher(seg).replaceAll(" "));
          }
          sb.append(matcher.group());
          segStart = matchEnd;
        }
        if (segStart < snippetText.length()) {
          String seg = snippetText.substring(segStart, snippetText.length());
          sb.append(discardPattern.matcher(seg).replaceAll(" "));
        }
        snippetText = sb.toString();
      }
      snippetText = translatePattern.matcher(snippetText).replaceAll(" ");
      snippetText = whitespacePattern.matcher(snippetText).replaceAll(" ");
      if (lowerCase) {
        snippetText = snippetText.toLowerCase();
      }
      return snippetText;
    }

    public void processSnippet(RelationMentionSnippets snippets, WebSnippet snippet)
    {
      String cleaned = cleanSnippet(snippet);
      snippet.setText(cleaned);
      if (nextHandler != null) {
        nextHandler.processSnippet(snippets, snippet);
      }
    }

  }
  
  public static class WebSnippetFilterer extends ChainedWebSnippetHandler
  {
    Set<RelationMentionSnippets.QueryType> matchQueryTypes;

    public WebSnippetFilterer(WebSnippetHandler nextHandler) {
      super(nextHandler);
    }

    @Override
    public void init(Properties props) {
      super.init(props);
      String queryTypesStr = props.getProperty("kbp.websnippets.filter.queryTypes");
      if (queryTypesStr != null) {
        String[] queryTypes = queryTypesStr.split(",");
        matchQueryTypes = EnumSet.noneOf(RelationMentionSnippets.QueryType.class);
        for (String queryType:queryTypes) {
          queryType = queryType.trim();
          matchQueryTypes.add(RelationMentionSnippets.QueryType.valueOf(queryType));
        }
      }
    }

    public void startMention(RelationMentionSnippets snippets)
    {
      if (matchQueryTypes == null || matchQueryTypes.contains(snippets.getQueryType())) {
        if (nextHandler != null) {
          nextHandler.startMention(snippets);
        }
      }
    }

    public void finishMention(RelationMentionSnippets snippets)
    {
      if (matchQueryTypes == null || matchQueryTypes.contains(snippets.getQueryType())) {
        if (nextHandler != null) {
          nextHandler.finishMention(snippets);
        }
      }
    }

    public void processSnippet(RelationMentionSnippets snippets, WebSnippet snippet)
    {
      if (matchQueryTypes == null || matchQueryTypes.contains(snippets.getQueryType())) {
        if (nextHandler != null) {
          nextHandler.processSnippet(snippets, snippet);
        }
      }
    }
  }

  public static class WebSnippetCacher extends AbstractWebSnippetHandler {
    String cacheDir;
    Map<String, EntityType> entityTypes;
    StanfordCoreNLP splitter;
    StanfordCoreNLP full;
    int minSentenceLength;
    Counter<String> fileIndex;
    boolean testMode;
    int docCount;
    int queriesSkipped;
    int queriesValid;
    
    /** If false, use our own custom serialization */
    public static final boolean USE_GENERIC_SERIALIZATION = false;
    
    public WebSnippetCacher(String cacheDir, Map<String, EntityType> entityTypes, boolean testMode, Properties props) {
      this.cacheDir = cacheDir;
      this.entityTypes = entityTypes;
      this.testMode = testMode;
      this.docCount = 0;
      this.queriesSkipped = 0;
      this.queriesValid = 0;
      
      props.setProperty("annotators", "tokenize, ssplit");
      splitter = new StanfordCoreNLP(props);
      props.setProperty("annotators", "pos, lemma, ner, regexner, parse, dcoref");
      full = new StanfordCoreNLP(props, false);
      
      minSentenceLength = Integer.parseInt(props.getProperty("kbp.websnippets.minsent", "15"));
      fileIndex = new ClassicCounter<String>();
    }

    @Override
    public void processSnippet(RelationMentionSnippets snippets, WebSnippet snippet) {
      snippets.add(snippet);
    }
    
    public void finishMention(RelationMentionSnippets snippets) {
      //
      // in testMode we only care about NEUtral queries
      // in ! testMode (i.e., train entities) we only care about POS queries. we extract both positive and negative queries from these sents
      //
      if(testMode){
        if(snippets.queryType != RelationMentionSnippets.QueryType.NEU) return;
      } else {
        if(snippets.queryType != RelationMentionSnippets.QueryType.POS) return;
      }
      
      // find out type of this entity (PER or ORG)
      String normName = StringEscapeUtils.unescapeHtml(snippets.entityName);
      EntityType myType = entityTypes.get(normName);
      if(myType == null){
        queriesSkipped ++;
        // this may happen if snippets are generated using an older version of the KB
        Log.severe("Unknown entity type for: " + snippets.entityName);
        return;
      } else {
        queriesValid ++;
      }
      
      // save only sentences for relations relevant to my type
      if((myType == EntityType.ORGANIZATION && snippets.slotName.startsWith("per:")) ||
         (myType == EntityType.PERSON && snippets.slotName.startsWith("org:"))){
        Log.severe("Discarding relation " + snippets.slotName + " due to incompatibility with entity type " + myType);
        return;
      }
      
      // convert snippets to actual sentences
      List<CoreMap> sentences = new ArrayList<CoreMap>();
      Log.severe("SNIPPETS for " + myType + ":" + snippets.entityName);
      for(WebSnippet snippet: snippets.snippets){
        Log.severe("\t" + snippet.text);
        snippetToSentences(snippet, sentences);
      }
      
      // skip pre-processing if there are no sentences
      if(sentences.size() == 0) return;
      docCount ++;
      
      // full processing, including coref
      for(CoreMap s: sentences){
        Log.fine("SENT: " + Utils.sentenceToMinimalString(s));     
      }
      Log.fine("Parsing the above sentences...");
      Annotation corpus = new Annotation("");
      corpus.set(SentencesAnnotation.class, sentences);
      boolean successfulPipeline = false;
      
      try {
        full.annotate(corpus);
        successfulPipeline = true;
      } catch(Exception e){
        e.printStackTrace();
        Log.severe("Exception above caught on the following sentences:");
        for(CoreMap sentence: corpus.get(SentencesAnnotation.class)){
          Log.severe(Utils.sentenceToMinimalString(sentence));
        }
        Log.severe("WARNING: pipeline exception caught for entity " + snippets.entityName + ". Continuing...");
      }
      
      if(successfulPipeline) {
        // create AntecedentAnnotation, based on the coref graph
        AntecedentGenerator antGen = new AntecedentGenerator(snippets.entityName, 100); // last param not actually used in this context
        antGen.findAntecedents(corpus);

        String myCacheDir = SentenceCacher.makeCacheDirNameAndCreate(cacheDir, snippets.entityName, myType);
        Log.fine("Saving cache for entity " + snippets.entityName + " to directory " + myCacheDir);
        String serFn = SentenceCacher.makeCacheFileName(cacheDir, snippets.entityName, myType, "cache");
        String debFn = SentenceCacher.makeCacheFileName(cacheDir, snippets.entityName, myType, "debug");
        String cstFn = SentenceCacher.makeCacheFileName(cacheDir, snippets.entityName, myType, "custom");
        String fileKey = myType.toString() + ":" + snippets.entityName;
        int fileCount = (int) fileIndex.getCount(fileKey);
        Log.fine("FILE COUNT for " + fileKey + " is " + fileCount);
        serFn = serFn + "." + fileCount;
        debFn = debFn + "." + fileCount;
        cstFn = cstFn + "." + fileCount;

        try {
          if(USE_GENERIC_SERIALIZATION){
            IOUtils.writeObjectToFile(sentences, new File(serFn));
            PrintStream os = new PrintStream(new FileOutputStream(debFn));
            for(CoreMap sent: sentences){
              SentenceCacher.saveSentenceDebug(os, sent);
            }
            os.close();
          } else {
            AnnotationSerializer cas = new KBPAnnotationSerializer(true, true);
            FileOutputStream os = new FileOutputStream(cstFn);
            cas.save(corpus, os); // custom serialization
            os.close();
          }
          fileIndex.incrementCount(fileKey);
        } catch (IOException e) {
          System.err.println("ERROR: cannot save sentences for entity " + snippets.entityName);
          e.printStackTrace();
          throw new RuntimeException(e);
        }
      }
    }
    
    /**
     * Splits the snippet into one or more sentences and adds the "meaty" ones to sentences
     * @param snippet
     * @param sentences
     */
    private boolean snippetToSentences(WebSnippet snippet, List<CoreMap> sentences) {
      // make sure it doesn't come from an InfoBox! It is FORBIDDEN to use infoboxes in KBP!
      if(snippet.text.toLowerCase().contains("infobox")){
        System.err.println("Found InfoBox! Skipping: " + snippet.text);
        return false;
      }
      
      String normed = snippet.text;
      // remove dates posted at the beginning of snippet. irrelevant here
      normed = normed.replaceAll("^[A-Z][a-z][a-z]\\s+\\d\\d?\\s*,\\s*[12]\\d\\d\\d", "");
      // remove ... prefixes
      normed = normed.replaceAll("^\\s*\\.\\.\\.+", "");
      // convert ... to .? this means we will break snippets into many sentences...
      // normed = normed.replaceAll("\\.\\.\\.+", ".");
      Annotation corpus = new Annotation(normed);
      splitter.annotate(corpus);
      
      for(CoreMap s: corpus.get(SentencesAnnotation.class)) {
        if(s.get(TokensAnnotation.class).size() >= minSentenceLength){
          sentences.add(s);
        }
      }
      
      return true;
    }
  }

  public static class WebSnippetWriter extends AbstractWebSnippetHandler
  {
    PrintStream out;
    boolean outputMentionsWithSnippetsOnly = false;  // Option to only output entries for relation mentions with snippets

    public WebSnippetWriter(PrintStream out) {
      this.out = out;
    }

    @Override
    public void init(Properties props) {
      super.init(props);
      outputMentionsWithSnippetsOnly = Boolean.parseBoolean(props.getProperty("kbp.websnippets.output.outputMentionsWithSnippetsOnly",
              Boolean.toString(outputMentionsWithSnippetsOnly)));
    }

    public void processSnippet(RelationMentionSnippets snippets, WebSnippet snippet)
    {
      snippets.add(snippet);
    }

    public void finishMention(RelationMentionSnippets snippets)
    {
      if (!outputMentionsWithSnippetsOnly || snippets.snippets.size() > 0) {
        out.println(snippets.toString());
      }
      out.flush();
    }

    public void finish()
    {
      out.flush();
    }
  }

  // Annotates and saves annotated snippets to file
  public static class WebSnippetAnnotator extends WebSnippetProcessor.AbstractWebSnippetHandler
  {
    StanfordCoreNLP pipeline;
    ObjectOutputStream oos;
    ObjectInputStream pois;
    int totalCachedRead = 0;
    TokenizerPostProcessorAnnotator tokenizerAnnotator;

    public WebSnippetAnnotator(StanfordCoreNLP pipeline, String filename) {
      this.pipeline = pipeline;
      tokenizerAnnotator = new TokenizerPostProcessorAnnotator();
      try {
        this.oos = new ObjectOutputStream(new BufferedOutputStream(IOUtils.getFileOutputStream(filename)));
      } catch (IOException ex) {
        throw new RuntimeException(ex);
      }
    }

    public WebSnippetAnnotator(StanfordCoreNLP pipeline, String filename, String partialCacheFile) {
      this(pipeline, filename);
      if (partialCacheFile != null) {
        System.err.println("Using partial cached file: " + partialCacheFile);
        try {
          this.pois = new ObjectInputStream(new BufferedInputStream(IOUtils.getFileInputStream(partialCacheFile)));
        } catch (IOException ex) {
          throw new RuntimeException(ex);
        }
      }
    }

    private Pair<String,List<Annotation>> getNextCached()
    {
      try {
        Object obj = pois.readObject();
        String header = ErasureUtils.<String> uncheckedCast(obj);
        obj = pois.readObject();
        List<Annotation> annotations = ErasureUtils.<List<Annotation>> uncheckedCast(obj);
        totalCachedRead++;
        if (totalCachedRead == 1) {
          System.err.println("Read from cached successfully");
        }
        return new Pair<String, List<Annotation>>(header, annotations);
      } catch (Exception ex) {
        // Error: WARN and reset pois
        System.err.println("WARNING: Error getting next cached entry: " + ex);
        System.err.println("WARNING: Aborting reading from cached after " + totalCachedRead + " read");
        try {
          pois.close();
        } catch (IOException ex2) {
          System.err.println("WARNING: Error closing cache reader: " + ex2);
        }
        pois = null;
        return null;
      }
    }

    private static final Pattern emSegment = Pattern.compile("<em>(.*?)</em>");
    public void finishMention(RelationMentionSnippets snippets)
    {
      String header = snippets.getHeaderF0();
      List<Annotation> annotations = null;
      if (pois != null) {
        Pair<String, List<Annotation>> p = getNextCached();
        if (p != null) {
          if (header.equals(p.first) && snippets.snippets.size() == p.second().size()) {
            annotations = p.second();
          } else {
            System.err.println("WARNING: Cached entry does not matched expected");
            System.err.println("WARNING:   expected " + snippets.snippets.size() + " for " + header);
            System.err.println("WARNING:   got " + p.second().size() + " for " + p.first());
          }
        }
      }
      if (annotations == null) {
        annotations = new ArrayList<Annotation>(snippets.snippets.size());
        for (WebSnippet snippet: snippets.snippets) {
          // Mark the position of <em> </em>
          String snippetText = snippet.getText();
          List<IntPair> emOffsets = MultiWordStringMatcher.findOffsets(emSegment, snippetText);
          // String out <em> </em> marks
          Matcher matcher = emSegment.matcher(snippetText);
          snippetText = matcher.replaceAll("    $1     ");
          Annotation annotation = new Annotation(snippetText);
          annotation.set(KBPAnnotations.MarkedPositionsAnnotation.class, emOffsets);
          tokenizerAnnotator.annotate(annotation);

          // For now, just have one sentence
          List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
          Annotation sentence = new Annotation(snippetText);
          sentence.set(CoreAnnotations.CharacterOffsetBeginAnnotation.class, 0);
          sentence.set(CoreAnnotations.CharacterOffsetEndAnnotation.class, snippetText.length());
          sentence.set(CoreAnnotations.TokensAnnotation.class, tokens);
          sentence.set(CoreAnnotations.TokenBeginAnnotation.class, 0);
          sentence.set(CoreAnnotations.TokenEndAnnotation.class, tokens.size());
          List<CoreMap> sentences = new ArrayList<CoreMap>(1);
          sentences.add(sentence);
          annotation.set(CoreAnnotations.SentencesAnnotation.class, sentences);

          // Do rest of pipeline
          pipeline.annotate(annotation);
          annotations.add(annotation);
        }
      }
      try {
        oos.writeObject(header);
        oos.writeObject(annotations);
        oos.reset();
      } catch (IOException ex) {
        throw new RuntimeException(ex);
      }
    }

    public void processSnippet(RelationMentionSnippets snippets, WebSnippet snippet)
    {
      snippets.add(snippet);
    }

    public void finish() {
      try {
        oos.flush();
        oos.close();
        if (pois != null) {
          pois.close();
        }
      } catch (IOException ex) {
        throw new RuntimeException(ex);
      }
    }
  }
  
  public static void usage()
  {
    System.err.println("java edu.stanford.nlp.kbp.webqueries.WebSnippetProcessor");
    List<String> validCommands = new ArrayList<String>(cmdMap.keySet());
    Collections.sort(validCommands);
    for (String cmd:validCommands) {
      System.err.println("     " + cmd + " - " + cmdMap.get(cmd));
    }
    System.exit(-1);
  }

  /**
   * Run using: -props kbp.properties -cmd <YOUR COMMAND>
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception {
    Properties properties = StringUtils.argsToProperties(args);
    Log.setLevel(Log.stringToLevel(properties.getProperty("logLevel", "INFO")));
    
    String cmd = properties.getProperty("cmd");
    Set<String> validCommands = cmdMap.keySet();
    if (cmd == null) {
      System.err.println("Please provide -cmd");
      usage();
    } else if (!validCommands.contains(cmd)) {
      System.err.println("Unknown cmd: " + cmd);
      System.err.println("Valid commands are " + validCommands);
      usage();
    }

    WebSnippetProcessor processor = new WebSnippetProcessor();
    String snippetFormatVersion = properties.getProperty("kbp.websnippets.format.version");
    if (snippetFormatVersion != null) {
      int version = Integer.parseInt(snippetFormatVersion);
      processor.formatVersion = version;
    }
    String directory = properties.getProperty("kbp.websnippets.dir");
    String cacheDir = properties.getProperty("kbp.websnippets.cache");
    String patternStr = properties.getProperty("kbp.websnippets.pattern");
    Pattern pattern = (patternStr != null)? Pattern.compile(patternStr):null;
    if ("printStats".equals(cmd)) {
      processor.printStats(directory, pattern, properties);
    } else if ("cleanSnippets".equals(cmd)) {
      processor.cleanSnippets(directory, pattern, properties);
    } else if ("printSnippets".equals(cmd)) {
      processor.printSnippets(directory, pattern, properties);
    } else if ("saveAnnotations".equals(cmd)) {
      String infile = properties.getProperty("kbp.websnippets.input.file");
      String cachedfile = properties.getProperty("kbp.websnippets.annotation.file.saved");
      String outfile = properties.getProperty("kbp.websnippets.annotation.file");
      StanfordCoreNLP pipeline = new StanfordCoreNLP(properties, false);
      processor.annotateSnippets(pipeline, infile, outfile, cachedfile);
      System.err.println(pipeline.timingInformation());
    } else if ("printAnnotations".equals(cmd)) {
      String file = properties.getProperty("kbp.websnippets.annotation.file");
      processor.printAnnotatedSnippets(file);
    } else if ("printSamples".equals(cmd)) {
      processor.printSamples(directory, pattern, properties);
    } else if("toCacheTest".equals(cmd)){
      // construct a map from entity name to entity type (we need this to understand what and where to save the cache)
      List<KBPEntity> mentions = TaskXMLParser.parseQueryFile(properties.getProperty("kbp.websnippets.entitytypes"));
      Map<String, EntityType> entityTypes = new HashMap<String, EntityType>();
      for(KBPEntity m: mentions) entityTypes.put(m.name, m.type);
      // converts web snippets into the the cache format that KBPReader understands
      processor.snippetsToCacheTest(directory, cacheDir, entityTypes, properties);
    } else if("toCacheTrain".equals(cmd)){
      List<KBPEntity> mentions = readKnowledgeBaseEntities(properties.getProperty("kbp.inputkb"), properties);
      Log.severe("Found " + mentions.size() + " entities in the KB.");
      Map<String, EntityType> entityTypes = new HashMap<String, EntityType>();
      for(KBPEntity m: mentions) entityTypes.put(m.name, m.type);
      processor.snippetsToCacheTrain(directory, cacheDir, entityTypes, properties);
    } else {
      throw new RuntimeException("Unknown command: " + cmd);
    }
/*
    WebSnippetProcessor processor = new WebSnippetProcessor();
    handler.init(properties);
    processor.processSnippets(directory, ".*ee_query.*", handler);
    handler.finish(); */
  }
  
  /** Reads all entities in our KB 
   * @throws ParserConfigurationException 
   * @throws SAXException 
   * @throws IOException */
  private static List<KBPEntity> readKnowledgeBaseEntities(String path, Properties props) throws IOException, SAXException, ParserConfigurationException {
    KBPDomReader domReader = new KBPDomReader(props);
    Map<KBPEntity, List<KBPSlot>> entitySlotValues = domReader.parse(path);
    return new ArrayList<KBPEntity>(entitySlotValues.keySet());
  }
}
