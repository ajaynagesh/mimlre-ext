package edu.stanford.nlp.kbp.temporal;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.logging.Level;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.PrintStream;
import java.io.Serializable;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.queryParser.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.SimpleFSDirectory;
import org.apache.lucene.util.Version;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.slotfilling.FeatureFactory;
import edu.stanford.nlp.kbp.slotfilling.KBPTrainer;

import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.kbp.slotfilling.common.DatumAndMention;
import edu.stanford.nlp.kbp.slotfilling.common.EntityType;
import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.KBPSlot;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.MinimalDatum;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.kbp.slotfilling.common.TemporalRelationMention;
import edu.stanford.nlp.kbp.slotfilling.common.Utils;

import edu.stanford.nlp.kbp.slotfilling.distantsupervision.KBPReader;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.TemporalSentenceExtractor;
import edu.stanford.nlp.kbp.slotfilling.index.LucenePipelineCacher;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.NumberAnnotator;
import edu.stanford.nlp.pipeline.QuantifiableEntityNormalizingAnnotator;
import edu.stanford.nlp.pipeline.RegexNERAnnotator;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.OpenAddressCounter;
import edu.stanford.nlp.time.SUTime;
import edu.stanford.nlp.time.TimeAnnotations.TimexAnnotations;
import edu.stanford.nlp.time.TimeAnnotator;
import edu.stanford.nlp.time.TimeExpression;
import edu.stanford.nlp.util.CollectionUtils;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.StringUtils;

public class FindTemporalExpressionsWiki implements Serializable {

  class FreeBaseTemporalArticle implements Serializable {
    private static final long serialVersionUID = -8285276969817133954L;

    String name;
    String wikiURL;
    String title;
    // key - label, Pair is the start and end dates
    Map<String, Pair<String, String>> temporalInfo;

    public FreeBaseTemporalArticle() {
      temporalInfo = new HashMap<String, Pair<String, String>>();
    }
  };

  public enum StartOrEnd {
    START, END
  };

  class TemporalInfo implements Serializable {

    private static final long serialVersionUID = 1L;
    CoreMap sentence;
    String entity;
    String dateValue;
    String label;
    StartOrEnd startEnd;

    public TemporalInfo(String entity, String label, CoreMap sentence, String date, StartOrEnd se) {
      this.entity = entity;
      this.label = label;
      this.sentence = sentence;
      this.dateValue = date;
      this.startEnd = se;
    }
  };

  void readGoogleNgrams(Properties props) throws Exception {

    Counter<String> startWords = IOUtils.readObjectFromFile(props.getProperty("temporal.startwordsfile"));
    Counter<String> endWords = IOUtils.readObjectFromFile(props.getProperty("temporal.endwordsfile"));

    Counter<String> allWords = Counters.union(startWords, endWords);
    Counter<String> countFromGoogleNGrams = new OpenAddressCounter<String>();

    String dir = "/scr/nlp/data/gale2/GoogleNgrams/2gms";
    for (File f : IOUtils.iterFilesRecursive(new File(dir), ".gz")) {
      for (String line : IOUtils.readLines(f, GZIPInputStream.class)) {
        String[] tokens = line.split("\t");
        String lowerCasePh = tokens[0].trim().toLowerCase();
        if (allWords.containsKey(lowerCasePh)) {
          System.out.println("found: " + tokens[0] + " with count " + tokens[1]);
          countFromGoogleNGrams.incrementCount(lowerCasePh, Double.parseDouble(tokens[1]));
        }
      }
      System.out.println("done reading " + f);
    }

    String dir2 = "/scr/nlp/data/gale2/GoogleNgrams/3gms";
    for (File f : IOUtils.iterFilesRecursive(new File(dir2), ".gz")) {
      for (String line : IOUtils.readLines(f, GZIPInputStream.class)) {
        String[] tokens = line.split("\t");
        String lowerCasePh = tokens[0].trim().toLowerCase();
        if (allWords.containsKey(lowerCasePh)) {
          System.out.println("found: " + tokens[0] + " with count " + tokens[1]);
          countFromGoogleNGrams.incrementCount(lowerCasePh, Double.parseDouble(tokens[1]));
        }
      }
      System.out.println("done reading " + f);
    }

    for (String line : IOUtils.readLines(new File("/scr/nlp/data/gale2/GoogleNgrams/1gms/vocab"))) {
      String[] tokens = line.split("\t");
      String lowerCasePh = tokens[0].trim().toLowerCase();
      if (allWords.containsKey(lowerCasePh)) {
        System.out.println("found: " + tokens[0] + " with count " + tokens[1]);
        countFromGoogleNGrams.incrementCount(lowerCasePh, Double.parseDouble(tokens[1]));
      }
    }
    System.out.println("done reading the 1 gram vocab file");
    IOUtils.writeObjectToFile(countFromGoogleNGrams, "ngramsCountFromGoogle.ser");
  }


  @SuppressWarnings("deprecation")
  void makePosNegConnotationWords(Properties props) throws Exception {

    props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
    StanfordCoreNLP pipeline = new StanfordCoreNLP(props, false);
    pipeline.addAnnotator(new NumberAnnotator(false));
    pipeline.addAnnotator(new QuantifiableEntityNormalizingAnnotator(false, false));
    pipeline.addAnnotator(new TimeAnnotator());

    String indexPath = "/scr/nlp/data/tackbp2010/indices/lr_en_100622_2000_Index_Cached";
    QueryParser parser;
    IndexSearcher searcher;
    Directory directory = new SimpleFSDirectory(new File(indexPath));
    searcher = new IndexSearcher(directory);
    StandardAnalyzer analyzer = new StandardAnalyzer(Version.LUCENE_CURRENT);
    parser = new QueryParser(Version.LUCENE_CURRENT, "title", analyzer);
    parser.setDefaultOperator(QueryParser.Operator.AND);

    Counter<String> posNgrams = new OpenAddressCounter<String>();
    Counter<String> negNgrams = new OpenAddressCounter<String>();

    // List<TemporalInfo> allTempObjs =
    // IOUtils.readObjectFromFile("allTempObjs.ser");

    // Log.setLevel(Level.parse(props.getProperty(Props.READER_LOG_LEVEL,
    // "SEVERE")));
    // KBPReader reader = new KBPReader(props, false, true);
    // reader.setLoggerLevel(Level.SEVERE);

    int numMatchingDocs = 0;
    String freebaseFile = props.getProperty("temporal.freeBaseFile");
    List<FreeBaseTemporalArticle> allArticles = IOUtils.readObjectFromFile(freebaseFile);
    System.out.println("size of allarticles is  " + allArticles.size());
    int minNumDatums = props.getProperty("temporal.minNumDatums") != null ? Integer.parseInt(props.getProperty("temporal.minNumDatums")) : 0;
    int maxNumDatums = props.getProperty("temporal.maxNumDatums") != null ? Integer.parseInt(props.getProperty("temporal.maxNumDatums")) : Integer.MAX_VALUE;
    int numDatums = 0;
    System.out.println("minNumDatums: " + minNumDatums + " and maxNumDatums: " + maxNumDatums);
    for (FreeBaseTemporalArticle a : allArticles) {
      numDatums++;
      if (numDatums < minNumDatums)
        continue;
      if (numDatums >= maxNumDatums) {
        System.out.println("Reached maximum number of datums!");
        break;
      }
      if (numDatums % 1000 == 0)
        System.out.println(numDatums);

      if (a.title == null)
        continue;

      Map<String, Pair<String, String>> temporalInfo = a.temporalInfo;
      if (temporalInfo.size() == 0)
        continue;

      String title = a.title.replaceAll("_", " ");
      if (title != null) {
        title = QueryParser.escape(title);
        Query q = parser.parse(title);
        TopDocs docs = searcher.search(q, 10);
        if (docs.scoreDocs == null || docs.scoreDocs.length == 0) {
          continue;
        }
        ScoreDoc scoredoc = docs.scoreDocs[0];
        int docId = scoredoc.doc;
        Document doc = searcher.doc(docId);
        if (doc == null) {
          Log.fine("Doc " + docId + " not found");
        } else {
          CoreMap annotatedText = LucenePipelineCacher.getAnnotationFromDoc(doc);
          List<CoreMap> sentences = annotatedText.get(SentencesAnnotation.class);

          for (CoreMap sentence : sentences) {

            List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
            if (tokens.size() > 200)
              continue;
            String sentenceStr = Utils.sentenceToMinimalString(sentence);

            Annotation aa = new Annotation(sentenceStr);

            try {
              pipeline.annotate(aa);
            } catch (Exception e) {
              continue;
            }

            List<CoreMap> cms = aa.get(TimexAnnotations.class);
            for (CoreMap cm : cms) {

              TimeExpression tm = cm.get(TimeExpression.Annotation.class);
              SUTime.Temporal t = tm.getTemporal();
              int beginToken = cm.get(CoreAnnotations.TokenBeginAnnotation.class);
              int endToken = cm.get(CoreAnnotations.TokenEndAnnotation.class);
              String dateVal = t.getTimexValue();
              List<String> dateNormalizedTokens = normalizeDateTokens(tokens, beginToken, endToken, dateVal);
              // List<Span> entitySpan = new ArrayList<Span>();
              // TemporalSentenceExtractor.matchSlotInSentence(title,
              // dateNormalizedTokens, title.split("\\s+"),
              // entitySpan);

              // System.out.println("sentence is " + sentenceStr +
              // " and title is " + title + " and span is "
              // + StringUtils.join(entitySpan));

              for (Map.Entry<String, Pair<String, String>> en : temporalInfo.entrySet()) {

                if (!en.getValue().first().equals("--")) {
                  String startDate = en.getValue().first();
                  List<Span> startDateSpans = new ArrayList<Span>();
                  if (TemporalSentenceExtractor.matchSlotInSentence(startDate, dateNormalizedTokens, new String[] { startDate }, startDateSpans)) {
                    // System.out.println("\n matched slot date " + startDate +
                    // " for sentence " + sentenceStr + " with spans "
                    // + StringUtils.join(startDateSpans));
                    Counters.addInPlace(posNgrams, getNGrams(dateNormalizedTokens, startDateSpans));
                    // foundSentences.add(aa.get(CoreAnnotations.SentencesAnnotation.class).get(0));
                  }
                }
                if (!en.getValue().second().equals("--")) {
                  List<Span> endDateSpans = new ArrayList<Span>();
                  String endDate = en.getValue().second();

                  if (TemporalSentenceExtractor.matchSlotInSentence(endDate, dateNormalizedTokens, new String[] { endDate }, endDateSpans)) {
                    // foundSentences.add(aa.get(CoreAnnotations.SentencesAnnotation.class).get(0));
                    Counters.addInPlace(negNgrams, getNGrams(dateNormalizedTokens, endDateSpans));
                  }
                }

              }
              numMatchingDocs++;
              // for (Fieldable field : doc.getFields()) {
              // if (field.name().equals("title"))
              // System.out.println(field.name() + doc.get(field.name()));

              // }

            }
          }
        }
      }
    }
    System.out.println("num documents matched " + numMatchingDocs);
    BufferedWriter w = new BufferedWriter(new FileWriter("posWords.txt_" + minNumDatums + "-" + maxNumDatums));
    Counters.retainAbove(posNgrams, 3);
    w.write(Counters.toSortedString(posNgrams, posNgrams.size(), "%1$s:%2$f", "\n"));
    w.close();

    BufferedWriter w2 = new BufferedWriter(new FileWriter("negWords.txt_" + minNumDatums + "-" + maxNumDatums));
    Counters.retainAbove(negNgrams, 3);
    w2.write(Counters.toSortedString(negNgrams, negNgrams.size(), "%1$s:%2$f", "\n"));
    w2.close();

    // System.out.println("\nNum matching docs " + numMatchingDocs +
    // " out of " + numAllDocs);
    // IOUtils.writeObjectToFile(allTempObjs, "allTempObjs.ser");

  }

  public static List<String> normalizeDateTokens(List<CoreLabel> tokens, int beginToken, int endToken, String dateVal) {
    List<String> dateNormalizedTokens = new ArrayList<String>();

    for (int i = 0; i < tokens.size(); i++) {
      if (i == beginToken)
        dateNormalizedTokens.add(dateVal);
      else if (i > beginToken && i <= endToken)
        continue;
      else
        dateNormalizedTokens.add(tokens.get(i).word());
    }
    return dateNormalizedTokens;
  }

  static int k = 3;
  static List<String> stopWords = Arrays.asList("a", "an", ",", "-lrb-", "-rrb-", ":", ".", "the", "\"", "'", "january", "february", "march", "april", "may", "june", "july",
      "august", "september", "october", "november", "december", "-", "--", "``", "`", "-lcb-", "-rcb-", "=", ";", "and", "or", "has", "had", "have", "there", "where", "he",
      "their", "she", "it");
  static List<String> wholeStopWords = Arrays.asList("was", "is", "in", "on", "of");

  static public List<String> getNGrams(List<String> str) {
    List<String> lowerCaseStr = new ArrayList<String>();
    for (String s : str) {
      if (s == null)
        continue;
      s = s.toLowerCase();
      if (s.matches("\\W\\W*") || s.matches("[\\s\\d-][\\s\\d-]*") || stopWords.contains(s) || s.length() <= 1)
        continue;
      lowerCaseStr.add(s);
    }
    List<String> ngrams = new ArrayList<String>();
    List<List<String>> ngramsList = CollectionUtils.getNGrams(lowerCaseStr, 1, 3);
    for (List<String> ngramsL : ngramsList) {
      if (ngramsL != null) {
        if (ngramsL.size() > 0) {
          String ngram = StringUtils.join(ngramsL, " ");
          if (!wholeStopWords.contains(ngram))
            ngrams.add(ngram);
        }
      }
    }
    return ngrams;
  }

  static public Set<String> getNGrams(List<String> dateNormalizedTokens, List<Span> dateSpans) {

    Set<String> ngrams = new HashSet<String>();
    // for (Span eS : entitySpans) {
    // for (Span dS : dateSpans) {
    // if (eS.end() < dS.start()) {
    // ngrams.addAll(getNGrams(dateNormalizedTokens.subList(eS.end() + 1,
    // dS.start())));
    // ngrams.addAll(getNGrams(dateNormalizedTokens.subList(Math.max(eS.start()
    // - k, 0), eS.start())))String.CASE_INSENSITIVE_ORDER;
    // ngrams.addAll(getNGrams(dateNormalizedTokens.subList(dS.end() + 1,
    // Math.min(dS.end() + k,
    // dateNormalizedTokens.size()))));
    // } else {
    // ngrams.addAll(getNGrams(dateNormalizedTokens.subList(dS.end() + 1,
    // eS.start())));
    // ngrams.addAll(getNGrams(dateNormalizedTokens.subList(Math.max(dS.start()
    // - k, 0), dS.start())));
    // ngrams.addAll(getNGrams(dateNormalizedTokens.subList(eS.end() + 1,
    // Math.min(eS.end() + k,
    // dateNormalizedTokens.size()))));
    // }
    //
    // }
    // }
    for (Span dS : dateSpans) {
      ngrams.addAll(getNGrams(dateNormalizedTokens.subList(Math.max(dS.start() - k, 0), Math.min(dS.end() + k, dateNormalizedTokens.size()))));
    }
    return ngrams;
  }

  void readNGramsFiles() throws Exception {
    Counter<String> posWords = new OpenAddressCounter<String>();
    Counter<String> negWords = new OpenAddressCounter<String>();

    for (File p : IOUtils.iterFilesRecursive(new File("/home/sonalg/javanlp/"), Pattern.compile("posWords.txt_.*"))) {
      for (String line : IOUtils.readLines(p)) {
        int colonIndex = line.lastIndexOf(":");
        String ngram = line.substring(0, colonIndex).trim();
        double count = Double.parseDouble(line.substring(colonIndex + 1));
        posWords.incrementCount(ngram, count);
      }
    }

    for (File n : IOUtils.iterFilesRecursive(new File("/home/sonalg/javanlp/"), Pattern.compile("negWords.txt_.*"))) {
      for (String line : IOUtils.readLines(n)) {
        int colonIndex = line.lastIndexOf(":");
        String ngram = line.substring(0, colonIndex).trim();
        double count = Double.parseDouble(line.substring(colonIndex + 1));
        negWords.incrementCount(ngram, count);
      }
    }
    System.out.println("size of poswords is " + posWords.size() + " and top words are " + Counters.toSortedString(posWords, 10, "%1$s:%2$f", ";"));
    System.out.println("size of negwords is " + negWords.size() + " and top words are " + Counters.toSortedString(negWords, 10, "%1$s:%2$f", ";"));
    IOUtils.writeObjectToFile(posWords, "posWords.ser");
    IOUtils.writeObjectToFile(negWords, "negWords.ser");
  }

  void createRelationMention(Properties props) throws Exception {

    String[] relationFeatures = props.getProperty(Props.RELATION_FEATS).split(",\\s*");
    assert (relationFeatures != null && relationFeatures.length > 0);
    Log.severe("relationFeatures: " + StringUtils.join(relationFeatures));
    FeatureFactory rff = new FeatureFactory(relationFeatures);
    rff.setDoNotLexicalizeFirstArgument(true);

    // List<TemporalInfo> allTempObjs =
    // IOUtils.readObjectFromFile("allTempObjs.ser");

    Log.setLevel(Level.parse(props.getProperty(Props.READER_LOG_LEVEL, "SEVERE")));
    KBPReader reader = new KBPReader(props, false, true, false);
    reader.setLoggerLevel(Level.SEVERE);

    Map<KBPEntity, List<KBPSlot>> entitySlotValues = new HashMap<KBPEntity, List<KBPSlot>>();
    String freebaseFile = props.getProperty("temporal.freeBaseFile");
    List<FreeBaseTemporalArticle> allArticles = IOUtils.readObjectFromFile(freebaseFile);
    int numEn = 0;
    Counter<String> localDomainStats = new ClassicCounter<String>();
    int minNumDatums = props.getProperty("temporal.minNumDatums") != null ? Integer.parseInt(props.getProperty("temporal.minNumDatums")) : 0;
    int maxNumDatums = props.getProperty("temporal.maxNumDatums") != null ? Integer.parseInt(props.getProperty("temporal.maxNumDatums")) : Integer.MAX_VALUE;
    int numDatums = 0;

    PrintStream os = new PrintStream(props.getProperty(Props.WORK_DIR) + "/train/temporal_" + minNumDatums + "-" + maxNumDatums + ".datums");
    for (FreeBaseTemporalArticle a : allArticles) {
      numDatums++;
      if (numDatums < minNumDatums)
        continue;
      if (numDatums >= maxNumDatums) {
        Log.severe("Reached maximum number of datums!");
        break;
      }

      if (a.title == null)
        continue;
      String title = a.title.replaceAll("_", " ");
      KBPEntity e = new KBPEntity();
      e.name = title;
      e.type = EntityType.PERSON;
      e.id = Integer.toString(++numEn);
      List<KBPSlot> relMens = new ArrayList<KBPSlot>();

      Map<String, Pair<String, String>> temporalInfo = a.temporalInfo;
      if (temporalInfo.size() == 0)
        continue;

      for (Map.Entry<String, Pair<String, String>> en : temporalInfo.entrySet()) {
        if (!en.getValue().first().equals("--")) {
          KBPSlot r = new KBPSlot(title, Integer.toString(numEn), en.getValue().first(), en.getKey() + "-start");
          relMens.add(r);
          // times.add(en.getValue().first());
        }
        if (!en.getValue().second().equals("--")) {
          // times.add(en.getValue().second());
          KBPSlot r = new KBPSlot(title, Integer.toString(numEn), en.getValue().second(), en.getKey() + "-end");
          relMens.add(r);
        }
      }

      entitySlotValues.put(e, relMens);
      // System.out.println("\n\nbefore read entity");
      List<CoreMap> sentences = reader.readEntity(e, entitySlotValues, null, null, true);
      Log.severe("size of sentences after reading entity is " + sentences.size());
      Annotation corpus = new Annotation("");
      corpus.set(SentencesAnnotation.class, sentences);

      List<DatumAndMention> dms = reader.generateDatums(corpus, rff, null, localDomainStats);
      // save the datum object, discard RelationMention objects
      List<MinimalDatum> datumsOutput = new ArrayList<MinimalDatum>();
      for (DatumAndMention dm : dms) {

        datumsOutput.add(new MinimalDatum(
            dm.mention().getArg(0).getObjectId(),
            dm.mention().getArg(0).getType(),
            dm.mention().getArg(1).getType(),
            dm.mention().getNormalizedSlotValue(),
            dm.datum()));

      }

      // save datums to stream
      for (MinimalDatum datum : datumsOutput) {
        if (!(datum.datum() instanceof BasicDatum<?, ?>)) {
          throw new RuntimeException("Datums must be BasicDatums here! This should NOT happen...");
        }
        datum.saveDatum(os);
        numDatums++;
      }

    }
    IOUtils.writeObjectToFile(entitySlotValues, props.getProperty("temporal.entitySlotsValue") + "_" + minNumDatums + "-" + maxNumDatums);
    // List<RelationMention> relations =
    // reader.createPositiveAndNegativeRelations(Arrays.asList(e),
    // entitySlotValues,
    // sentence, knownSlots);
    // if (relations != null && relations.size() > 0) {
    //
    // sentence.set(RelationMentionsAnnotation.class, relations);
    //
    // List<CoreMap> sentences = reader.readEntity(e, entitySlotValues, null,
    // true);
    // Annotation corpus = new Annotation("");
    // corpus.set(SentencesAnnotation.class, sentences);
    //
    // }
  }

  void train(Properties props) throws Exception {
    File workDir = new File(props.getProperty(Props.WORK_DIR));
    assert (workDir.isDirectory());
    File trainDir = new File(workDir + File.separator + "train");
    KBPTrainer trainer = new KBPTrainer(props);
    Map<KBPEntity, List<KBPSlot>> entitySlotValues = IOUtils.readObjectFromFile(props.getProperty("temporal.entitySlotsValue"));

    double samplingRatio = PropertiesUtils.getDouble(props,
        Props.NEGATIVES_SAMPLE_RATIO,
        Constants.DEFAULT_NEGATIVES_SAMPLING_RATIO);

    if (!trainer.modelExists()) {
      // construct dataset and train
      List<File> trainDatumFiles = KBPTrainer.fetchFiles(trainDir.getAbsolutePath(), ".datums");

      File negFile = null;
      if(Constants.OFFLINE_NEGATIVES) {
        negFile = new File(trainDir + File.separator +
            "datums_" + (int) (100.0 * samplingRatio) + ".negatives");
        if(! negFile.exists()) {
          KBPTrainer.subsampleNegatives(trainDatumFiles, negFile, samplingRatio);
        }
      }

      Map<String, Set<String>> slotsByEntityId = extractSlotsById(entitySlotValues);
      trainer.setEntitySlotsById(slotsByEntityId);
      trainer.trainOneVsAll(trainDatumFiles, negFile);
    } else
      throw new Exception("trainer file exists");

  }

  Map<String, Set<String>> readSlotsByEntityId(Properties props) throws Exception {
    Map<KBPEntity, List<KBPSlot>> entitySlots = IOUtils.readObjectFromFile(props.getProperty("temporal.entitySlotsValue"));
    Map<String, Set<String>> slotsByEntityId = extractSlotsById(entitySlots);
    return slotsByEntityId;
  }

  Map<String, Set<String>> extractSlotsById(Map<KBPEntity, List<KBPSlot>> entitySlots) {
    Counter<String> slotStats = new ClassicCounter<String>();
    Map<String, Set<String>> slotsById = new HashMap<String, Set<String>>();
    for (KBPEntity ent : entitySlots.keySet()) {
      Set<String> mySlots = new HashSet<String>();
      Collection<KBPSlot> slots = entitySlots.get(ent);
      for (KBPSlot slot : slots) {
        mySlots.add(slot.slotName);
        slotStats.incrementCount(slot.slotName);
      }
      // System.err.println("SLOTS FOR ENTITY " + ent.id + ": " + mySlots);
      slotsById.put(ent.id, mySlots);
    }
    Log.severe("Slot stats in the KB: " + slotStats);
    return slotsById;
  }

  void onlineNer(RegexNERAnnotator regexAnnotator, List<CoreMap> sentences) {
    if (regexAnnotator != null) {
      Annotation corpus = new Annotation("");
      corpus.set(SentencesAnnotation.class, sentences);
      regexAnnotator.annotate(corpus);
    }
  }

  Log logger;
  Properties props;

  void learnFromAnnotations(Properties props) throws Exception {
    KBPReader reader = new KBPReader(props, false, true, false);
    PrintStream os = new PrintStream(new FileOutputStream(props.getProperty("test.output", "test_datums.dat")));
    Counter<String> labelStats = new ClassicCounter<String>();
    Counter<String> domainStats = new ClassicCounter<String>();
    String[] relationFeatures = props.getProperty(Props.RELATION_FEATS).split(",\\s*");
    FeatureFactory rff = new FeatureFactory(relationFeatures);
    rff.setDoNotLexicalizeFirstArgument(true);
    reader.parse(os, props.getProperty("test.input"), rff, labelStats, domainStats);
    os.close();
  }

  void writePredictedSlotValue(TemporalRelationMention tr) {

  }

  private static final long serialVersionUID = 1L;

  // read David's Freebase output temporal file at
  // /u/mcclosky/res/temporal/hog/freebase/tuples.txt
  void readFreeBaseTemporalArticles(String outFile) {
    try {
      String file = "/u/mcclosky/res/temporal/hog/freebase/tuples.txt";
      List<FreeBaseTemporalArticle> allArticles = new ArrayList<FreeBaseTemporalArticle>();
      FreeBaseTemporalArticle article = null;
      for (String line : IOUtils.readLines(file)) {
        line = line.trim();
        if (line.length() == 0)
          continue;
        if (line.startsWith("name")) {
          if (article != null) {
            allArticles.add(article);
          }
          article = new FreeBaseTemporalArticle();
          String[] tokens = line.split("\\s+");
          String name = tokens[1];
          article.name = name;
          // System.out.println("name is " + name);
        }
        if (line.startsWith("wikipedia")) {
          String[] tokens = line.split("\\s+");
          String wikiURL = tokens[1];
          String wikiTitle = wikiURL.substring(wikiURL.lastIndexOf("/") + 1);
          article.wikiURL = wikiURL;
          article.title = wikiTitle;
          // System.out.println("wikititle is " + wikiTitle);
        }
        if (line.startsWith("span")) {
          String[] tokens = line.split("\\s+");
          String spanLabel = tokens[1];
          String startDate = tokens[2];
          String endDate = tokens[3];
          // System.out.println("span is  " + spanLabel +
          // " and start-end dates are " + startDate + "\t" + endDate);
          Pair<String, String> info = new Pair<String, String>(startDate, endDate);
          article.temporalInfo.put(spanLabel, info);
        }
      }
      IOUtils.writeObjectToFile(allArticles, outFile);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  static public void main(String[] args) {
    try {
      FindTemporalExpressionsWiki f = new FindTemporalExpressionsWiki();
      // f.readFreeBaseTemporalArticles("freebasearticlesinfo.ser");
      // f.findWikiArticlesFromLuceneIndex("freebasearticlesinfo.ser");
      Properties props = StringUtils.argsToProperties(args);
      // f.makePosNegConnotationWords(props);
      // f.readNGramsFiles();
      f.readGoogleNgrams(props);
      // if (Boolean.parseBoolean(props.getProperty("temporal.train", "false")))
      // {
      // System.out.println("training!!");
      // f.train(props);
      // } else if
      // (Boolean.parseBoolean(props.getProperty("temporal.createDatums",
      // "false")))
      // f.createRelationMention(props);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
