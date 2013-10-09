package edu.stanford.nlp.kbp.slotfilling.distantsupervision;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.kbp.slotfilling.common.*;
import org.apache.lucene.store.NoSuchDirectoryException;
import org.xml.sax.SAXException;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.ExtractionObject;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreAnnotations.DocIDAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations.RelationMentionsAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations.TriggerAnnotation;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.slotfilling.FeatureFactory;
import edu.stanford.nlp.kbp.slotfilling.KBPTemporal;
import edu.stanford.nlp.kbp.slotfilling.MentionCompatibility;
import edu.stanford.nlp.kbp.slotfilling.SlotValidity;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.SlotMentionsAnnotation;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.SourceIndexAnnotation;
import edu.stanford.nlp.kbp.slotfilling.index.PipelineIndexExtractor;
import edu.stanford.nlp.kbp.slotfilling.index.IndexExtractor.ResultSortMode;
import edu.stanford.nlp.kbp.temporal.TemporalWithSpan;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.AnswerAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.BeginIndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.EndIndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationPipeline;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.pipeline.MorphaAnnotator;
import edu.stanford.nlp.pipeline.NumberAnnotator;
import edu.stanford.nlp.pipeline.POSTaggerAnnotator;
import edu.stanford.nlp.pipeline.PTBTokenizerAnnotator;
import edu.stanford.nlp.pipeline.QuantifiableEntityNormalizingAnnotator;
import edu.stanford.nlp.pipeline.RegexNERAnnotator;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.pipeline.WordsToSentencesAnnotator;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.time.TimeAnnotations.TimexAnnotations;
import edu.stanford.nlp.time.TimeAnnotator;
import edu.stanford.nlp.time.TimeExpression;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.trees.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.StringUtils;

public class KBPReader extends GenericDataSetReader {

  /** Parses the KBP knowledge base and extracts <entity, slot value> tuples */
  private final KBPDomReader domReader;

  /**
   * Fetches relevant sentences for a given entity from all our resources (i.e.,
   * index, web cache)
   */
  private EntitySentenceExtractor sentenceExtractor;

  /** Used to generate unique entity mention ids */
  private int entityMentionCount;

  /** Used to generate unique relation mention ids */
  private int relationMentionCount;

  /** Candidates must match one of these NE labels */
  private final SlotsToNamedEntities slotsToNamedEntities;

  /** Whether to match the NE labels for each slot, default is true */
  private final boolean matchSlotNE;
  /**
   * Histogram which counts how many entities we have for a given number of
   * found sentences
   */
  private Counter<Integer> entToSentHistogram;

  /** Counts positive examples per relation type */
  private Counter<String> relationExampleCount;

  /** Are we running in test mode, i.e., answering the test queries? */
  private final boolean testMode;

  /** if read the gold slot name and values from the gold responses file */
  private final boolean diagnosticMode;

  /** needed for diagnostics model */
  private final String goldResponsesFile;
  /**
   * We do the regex NER online. There are several errors that were fixed since
   * the last corpus caching
   */
  private final RegexNERAnnotator regexAnnotator;

  /**
   * If true, a given slot candidate can match multiple slots (e.g., the same
   * country can be country_of_birth and country_of_death
   */
  private final boolean allowMultipleSlotMatchesPerMention;

  /** Enforce slot candidates to match known NE labels for that slot */
  private final boolean enforceNeInTraining;

  /** Should we use Daume's domain adaptation trick? */
  private final boolean domainAdaptation;
  private final String domainAdaptationStyle;

  /** This pipeline used only if temporal == true */
  private AnnotationPipeline pipeline;

  boolean temporal = false;
  boolean useDocDate = true;
  private Map<String, String> docDatesMap;

  private Annotator sutime;

  /**
   * If true, accepted only a specific subset of dependency paths between
   * temporals and slots
   */
  private final boolean filterDepPaths;
  /**
   * Use the attached-to-governor-verb temporal filter (triggered only when
   * filterDepPaths == true)
   */
  private final boolean useAttachToGovernorVerb;

  private List<TriggerSeq> triggers;

  private boolean alternateDateHandling;

  private static final Pattern YEAROLD_REGEX = Pattern.compile("[1-9][0-9]?-year-old", Pattern.CASE_INSENSITIVE);
  private static final Pattern DDDASH_REGEX = Pattern.compile("[1-9][0-9]?-", Pattern.CASE_INSENSITIVE);
  public static final Pattern YEAR_REGEX = Pattern.compile("[12]\\d\\d\\d");
  private static final Set<String> RANGE_WORDS = new HashSet<String>(Arrays.asList(new String[] { "-", "--", "to" }));
  private static final String DEFAULT_VALID_POS = "^(NN|JJ)";

  public KBPReader(Properties props, boolean testMode, boolean temporal, boolean diagnosticMode) throws Exception {
    super(new StanfordCoreNLP(props), true, false, true);
    // note that we need to pass the super c'tor a StanfordCoreNLP object,
    // because GenericDataSetReader requires a parser
    // because of this, the annotator pool must be initialized by calling the
    // StanfordCoreNLP c'tor

    this.testMode = testMode;
    this.diagnosticMode = diagnosticMode;

    if (diagnosticMode == true && temporal == false)
      throw new RuntimeException(
          "diagnostic mode cannot be true if temporal mode is false. Not currently implemented for non-temporal mode.");
    if (diagnosticMode == true)
      this.goldResponsesFile = props.getProperty(Props.GOLD_RESPONSES, null);
    else
      this.goldResponsesFile = null;

    this.temporal = temporal;
    this.useDocDate = Boolean.parseBoolean(props.getProperty(Props.TEMPORAL_USEDOCDATE, "true"));
    entityMentionCount = 0;
    relationMentionCount = 0;

    domReader = new KBPDomReader(props);

    int indexSentencesPerEntity = 0;
    int webSentencesPerEntity = 0;
    if (testMode) {
      indexSentencesPerEntity = Integer.parseInt(props.getProperty(Props.TEST_SENTENCES_PER_ENTITY, "10000"));
      webSentencesPerEntity = Integer.parseInt(props.getProperty(Props.TEST_WEBSENTENCES_PER_ENTITY, "10000"));
    } else {
      indexSentencesPerEntity = Integer.parseInt(props.getProperty(Props.TRAIN_SENTENCES_PER_ENTITY, "10000"));
      webSentencesPerEntity = Integer.parseInt(props.getProperty(Props.TRAIN_WEBSENTENCES_PER_ENTITY, "10000"));
    }
    String cacheDir = props.getProperty(Props.SENTENCE_CACHE); // old sentence
    // cache from KBP
    // 2010
    String testCacheDir = props.getProperty(Props.TEST_CACHE);
    if (Constants.USE_OLD_CACHING) {
      assert (cacheDir != null);
      assert (testCacheDir != null);
    }
    logger.info("Using cache for test sentences: " + testCacheDir);
    boolean useWeb = Boolean.parseBoolean(props.getProperty(Props.USE_WEB, "false"));
    String webCacheDir = null;
    if (useWeb == true) {
      webCacheDir = props.getProperty(Props.READER_WEBCACHE);
      assert (webCacheDir != null);
      logger.info("Using web cache from: " + webCacheDir);
    }
    boolean useCache = false;
    if (testMode) {
      useCache = Boolean.valueOf(props.getProperty(Props.TEST_USEINDEXCACHE, "false"));
    } else {
      useCache = Boolean.valueOf(props.getProperty(Props.TRAIN_USEINDEXCACHE, "false"));
    }
    String indexCacheDir = null;
    if (useCache) {
      indexCacheDir = props.getProperty(Props.INDEX_CACHE_DIR);
    }
    ResultSortMode sortMode = ResultSortMode.NONE;
    if (testMode) {
      sortMode = ResultSortMode.valueOf(props.getProperty(Props.TEST_RESULT_SORT_MODE_PROPERTY, ResultSortMode.NONE
          .toString()));
    } else {
      sortMode = ResultSortMode.valueOf(props.getProperty(Props.TRAIN_RESULT_SORT_MODE_PROPERTY, ResultSortMode.NONE
          .toString()));
    }
    boolean useTemporalSentenceExtractor = props.getProperty(Props.KBP_TEMPORAL_SENTENCEEXTRACTOR) != null ? Boolean
        .parseBoolean(props.getProperty(Props.KBP_TEMPORAL_SENTENCEEXTRACTOR)) : false;

    try {
      if (Constants.USE_OLD_CACHING) {
        sentenceExtractor = new AllCacheSentenceExtractor(indexSentencesPerEntity, webSentencesPerEntity, cacheDir,
                webCacheDir, testCacheDir);
      } else if (useTemporalSentenceExtractor) {
        sentenceExtractor = new TemporalSentenceExtractor(indexSentencesPerEntity);
      } else {
        sentenceExtractor = new IndexAndWebCacheSentenceExtractor(
                indexSentencesPerEntity, webSentencesPerEntity,
                indexCacheDir, webCacheDir, sortMode, props);
      }
    } catch(RuntimeException e) {
      logger.severe("WARNING: could not initialize the KBP index from the property " + Props.INDEX);
      logger.severe("This might still work if caching is available, so I will try to continue.");
    }

    slotsToNamedEntities = new SlotsToNamedEntities(props.getProperty(Props.NERENTRY_FILE));
    matchSlotNE = Boolean.valueOf(props.getProperty(Props.MATCH_SLOTNE, "true"));

    entToSentHistogram = new ClassicCounter<Integer>();
    relationExampleCount = new ClassicCounter<String>();

    String triggerFile = props.getProperty(Props.TRIGGER_WORDS);
    if (triggerFile != null) {
      try {
        loadTriggerWords(triggerFile);
      } catch (IOException e) {
        logger.severe("Cannot load trigger words from " + triggerFile);
        throw new RuntimeException(e);
      }
    } else {
      triggers = null;
    }

    allowMultipleSlotMatchesPerMention = Boolean.valueOf(props.getProperty(Props.MULTI_MATCH, "true"));
    logger.severe("KBPReader.allowMultipleSlotMatchesPerMention = " + allowMultipleSlotMatchesPerMention);
    enforceNeInTraining = Boolean.valueOf(props.getProperty(Props.ENFORCE_NE, "true"));
    logger.severe("KBPReader.enforceNeInTraining = " + enforceNeInTraining);

    String mapping = props.getProperty(Props.REGEX_MAP);
    boolean ignoreCase = props.containsKey(Props.REGEX_IGNORE_CASE) ? Boolean.valueOf(props
        .getProperty(Props.REGEX_IGNORE_CASE)) : false;
    // System.err.println("regexAnnotator: " + ignoreCase + " " + mapping);
    // The expected behavior is that if we are ignoring case, then
    // only some pos tags trigger the regex ner.  If we are not
    // ignoring case, then all pos tags are allowed.
    String validPosTags = ((ignoreCase) ?
                           DEFAULT_VALID_POS :
                           null);
    regexAnnotator = new RegexNERAnnotator(mapping, ignoreCase, validPosTags);

    domainAdaptation = PropertiesUtils.getBool(props, Props.DOMAIN_ADAPT, false);
    domainAdaptationStyle = props.getProperty(Props.DOMAIN_ADAPT_STYLE, "all");

    filterDepPaths = PropertiesUtils.getBool(props, Props.TEMPORAL_FILTER_DEP_PATHS, true);
    useAttachToGovernorVerb = PropertiesUtils.getBool(props, Props.TEMPORAL_GOV_VERB_FILTER, true);

    alternateDateHandling = PropertiesUtils.getBool(props, Props.ALTERNATE_DATE_HANDLING);

    if (temporal) {
      try {
        KBPTemporal.readStartEndFiles(props);
        pipeline = new AnnotationPipeline();
        pipeline.addAnnotator(new PTBTokenizerAnnotator(false));
        pipeline.addAnnotator(new WordsToSentencesAnnotator(false));
        pipeline.addAnnotator(new POSTaggerAnnotator(false));
        pipeline.addAnnotator(new MorphaAnnotator(false));
        pipeline.addAnnotator(new NumberAnnotator(false));
        pipeline.addAnnotator(new QuantifiableEntityNormalizingAnnotator(false, false));
        // this is a hack; see the comments on the function
        docDatesMap = KBPTemporal.readDocumentDates();
        logger.severe("size of docdatesmap is " + docDatesMap.size());
        sutime = new TimeAnnotator();
      } catch (Exception e) {
        logger.severe("TEMPORAL is TRUE but can't load the pipeline.");
        e.printStackTrace();
        System.exit(-1);
      }
    }
  }

  public Map<KBPEntity, List<KBPSlot>> parseKnowledgeBase(String kbPath) throws Exception {
    return domReader.parse(kbPath);
  }

  private KBPEntity getEntityMentionFromQueryId(List<KBPEntity> mentions, String queryId) {
    for (KBPEntity m : mentions) {
      if (m.queryId.equals(queryId)) return m;
    }
    return null;
  }

  Map<KBPEntity, List<KBPSlot>> parseGoldFile(String queryFile, String goldFile) throws IOException, SAXException {
    Map<KBPEntity, List<KBPSlot>> map = new HashMap<KBPEntity, List<KBPSlot>>();
    List<KBPEntity> mentions = TaskXMLParser.parseQueryFile(queryFile);
    for (KBPEntity em : mentions) {
      map.put(em, new ArrayList<KBPSlot>());
      logger.severe("Loaded KBP entity: " + em);
    }
    for (String line : IOUtils.readLines(goldFile)) {
      String[] tokens = line.split("\\s+", 5);
      String queryId = tokens[0].trim();
      String slotName = tokens[1].trim().replaceAll("/", "SLASH");
      String docId = tokens[5].trim();
      String slotValue = tokens[6].trim();
      KBPEntity m = getEntityMentionFromQueryId(mentions, queryId);
      if (m == null) {
        continue;
      }
      KBPSlot rm = new KBPSlot(m.name, m.id, slotValue, slotName);
      rm.docid = docId;
      map.get(m).add(rm);
    }
    return map;
  }

  Map<KBPEntity, List<KBPSlot>> parseGoldDiagnosticFile(String queryFile, String goldFile) throws IOException,
      SAXException {
    Map<KBPEntity, List<KBPSlot>> map = new HashMap<KBPEntity, List<KBPSlot>>();
    List<KBPEntity> mentions = TaskXMLParser.parseQueryFile(queryFile);
    for (KBPEntity em : mentions) {
      map.put(em, new ArrayList<KBPSlot>());
      logger.severe("Loaded KBP entity: " + em);
    }
    for (String line : IOUtils.readLines(goldFile)) {
      String[] tokens = line.split("\\s+", 7);
      assert (tokens.length == 7);
      String queryId = tokens[0].trim();
      String slotName = tokens[1].trim().replaceAll("/", "SLASH");
      String docId = tokens[5].trim();
      String slotValue = tokens[6].trim();
      logger.fine("Read gold slot: " + queryId + " \"" + slotValue + "\"");
      KBPEntity m = getEntityMentionFromQueryId(mentions, queryId);
      if (m == null) {
        continue;
      }
      KBPSlot rm = new KBPSlot(m.name, m.id, slotValue, slotName);
      rm.docid = docId;
      map.get(m).add(rm);
    }
    logger.severe("Read all gold slots");
    return map;
  }

  public static Map<KBPEntity, List<KBPSlot>> parseQueryFile(String fn) throws IOException, SAXException {
    Map<KBPEntity, List<KBPSlot>> map = new HashMap<KBPEntity, List<KBPSlot>>();
    List<KBPEntity> mentions = TaskXMLParser.parseQueryFile(fn);
    for (KBPEntity em : mentions) {
      map.put(em, new ArrayList<KBPSlot>());
      Log.info("Loaded KBP entity: " + em);
    }
    Log.severe("Found " + map.keySet().size() + " entity queries.");
    return map;
  }

  @Override
  public Annotation read(String path) throws Exception {

    Annotation corpus = new Annotation("");
    List<CoreMap> allSentences = new ArrayList<CoreMap>();
    entToSentHistogram = new ClassicCounter<Integer>();
    relationExampleCount = new ClassicCounter<String>();
    ReadStats stats = new ReadStats();

    //
    // parse the knowledge base, get a map from entities to slots
    // fetch slot values for all entities in the KB
    //
    Map<KBPEntity, List<KBPSlot>> entitySlotValues = loadEntitiesAndSlots(path);

    //
    // for each entity:
    // - fetch sentences from the index
    // - extract matches of the entity of interest
    // - extract slot filler candidates
    // - construct relation datums, both positive (using fillers that match
    // known values) and negatives (all other fillers)
    //
    List<KBPEntity> sortedEntities = new ArrayList<KBPEntity>(entitySlotValues.keySet());
    Collections.sort(sortedEntities, new Comparator<KBPEntity>() {
      public int compare(KBPEntity o1, KBPEntity o2) {
        return o1.name.compareTo(o2.name);
      }
    });
    // extract sentences for each individual entity
    for (KBPEntity entity : sortedEntities) {
      List<CoreMap> sentences = readEntity(entity, entitySlotValues, new File(path), stats, false);
      allSentences.addAll(sentences);
    }

    // the actual corpus
    corpus.set(SentencesAnnotation.class, allSentences);

    // report some stats
    reportStats(stats, allSentences, entitySlotValues);
    logger.severe("KBPReader.read complete.");
    return corpus;
  }

  public Map<KBPEntity, List<KBPSlot>> loadEntitiesAndSlots(String path) throws Exception {
    Map<KBPEntity, List<KBPSlot>> entitySlotValues = null;

    // actual parsing
    if (!testMode) {
      logger.info("Parsing the XML KB from " + path);
      entitySlotValues = parseKnowledgeBase(path);
    } else if (testMode && diagnosticMode) {
      logger.info("Getting the gold slot values from the gold response file: " + goldResponsesFile
          + " and the query file " + path);
      // entitySlotValues = parseGoldFile(path, goldResponsesFile);
      entitySlotValues = parseGoldDiagnosticFile(path, goldResponsesFile);
    } else if (testMode) {
      logger.info("Test mode: parsing the query file " + path);
      entitySlotValues = parseQueryFile(path);
    } else {
      throw new Exception("cannot reach here!");
    }

    // some stats
    int tupleCount = 0;
    Set<KBPEntity> allEntities = entitySlotValues.keySet();
    logger.severe("Found " + allEntities.size() + " known entities in " + path);
    for (KBPEntity key : allEntities) {
      logger.fine("Found entity: " + key);
      Collection<KBPSlot> slots = entitySlotValues.get(key);
      tupleCount += slots.size();
    }
    logger.severe("Found " + tupleCount + " tuples for " + allEntities.size() + " entities in " + path);

    return entitySlotValues;
  }

  public void reportStats(ReadStats stats, List<CoreMap> allSentences, Map<KBPEntity, List<KBPSlot>> entitySlotValues) {
    //
    // report some stats
    //
    logger.info("Distribution of entities by the number of sentences found:");
    List<Integer> sentCounts = new ArrayList<Integer>(entToSentHistogram.keySet());
    Collections.sort(sentCounts);
    for (Integer sentCount : sentCounts) {
      logger.info("Entities with " + sentCount + " sentences: " + entToSentHistogram.getCount(sentCount));
    }
    logger.info("Total number of sentences in the corpus: " + allSentences.size());
    logger.info("Number of examples per slot type:");
    List<String> slotTypes = new ArrayList<String>(relationExampleCount.keySet());
    for (String slotType : slotTypes) {
      logger.info("Number of examples for slot type " + slotType + ": " + relationExampleCount.getCount(slotType));
      if (relationExampleCount.getCount(slotType) > 0 && stats.relationTuples.get(slotType) != null)
        logger.info("Tuples for slot type: " + slotType + ": " + setsToString(stats.relationTuples.get(slotType)));
    }

    logger.info("+/- relation counter: " + stats.positiveAndNegativeCounter);
    logger.info("+ relation counter: " + stats.positiveCounter);
    logger.info("Number of sentences processed: " + stats.sentenceCounter);
    logger.info("Number of entities processed: " + entitySlotValues.keySet().size() + " with " + stats.slotCount
        + " total slots.");
  }

  public List<CoreMap> readEntity(KBPEntity entity, Map<KBPEntity, List<KBPSlot>> entitySlotValues, File sourceFile,
      ReadStats stats, boolean runGenericPreprocessor) throws Exception {
    logger.fine("Searching for entity: " + entity);
    List<CoreMap> sentencesWithExamples = new ArrayList<CoreMap>();
    int myPositiveSlots = 0;

    // make sure all entities have a unique id: we need this for merging
    // mentions between the same entity and the same slot
    if (entity.id == null) {
      throw new RuntimeException("Found an entity without an id: " + entity);
    }

    // used only for diagnostic task
    Set<String> validDocIds = null;

    //
    // extract all sentences that contain this entity
    //
    List<KBPSlot> knownSlots = null;
    if (testMode && !diagnosticMode) {
      // cannot use any known slots during testing
      knownSlots = new ArrayList<KBPSlot>();
    } else {
      // we get here also with diagnosticMode == true
      knownSlots = new ArrayList<KBPSlot>(entitySlotValues.get(entity));

      if (diagnosticMode) {
        for (KBPSlot s : knownSlots) {
          if (validDocIds == null) validDocIds = new HashSet<String>();
          validDocIds.add(s.docid.trim());
        }
      }

      if (knownSlots == null) {
        knownSlots = new ArrayList<KBPSlot>();
      } else {
        Collections.sort(knownSlots, new Comparator<KBPSlot>() {
          @Override
          public int compare(KBPSlot o1, KBPSlot o2) {
            return o1.slotName.compareTo(o2.slotName);
          }
        });
      }
    }

    if (stats != null) stats.slotCount += knownSlots.size();

    Set<String> knownSlotsAsKeywords = PipelineIndexExtractor.slotKeywords(knownSlots, alternateDateHandling);
    String entityDescription = "'" + entity.name + "'";
    if (entity.queryId != null) {
      entityDescription += " (query ID: " + entity.queryId + ")";
    }
    logger.severe("Using following slots for entity " + entityDescription + ": " + knownSlotsAsKeywords);
    if (validDocIds == null)
      logger.severe("valid doc ids are null! Not to worry if you are not running KBP temporal diagnostic task");
    else
      logger.severe(("valid doc ids are " + StringUtils.join(validDocIds, ";")));
    List<CoreMap> sentences = sentenceExtractor.findSentences(entity, knownSlotsAsKeywords, sourceFile, testMode,
        validDocIds);

    logger.fine("Found " + sentences.size() + " sentences containing entity " + entity);
    entToSentHistogram.incrementCount(sentences.size() < 100 ? sentences.size() : 100);
    assert (knownSlots != null);
    if (stats != null) stats.sentenceCounter += sentences.size();

    //
    // some NER components run online for each sentence
    //
    for (CoreMap sentence : sentences)
      logger.finest("Sentence before onlineNER: " + Utils.sentenceToString(sentence));
    onlineNer(sentences);
    for (CoreMap sentence : sentences)
      logger.finest("Sentence after onlineNER: " + Utils.sentenceToString(sentence));

    //
    // construct all versions of the entity name, e.g., for persons remove
    // middle name
    //
    List<String> alternateNames = new ArrayList<String>();
    if (entity.type == EntityType.PERSON && Constants.EXACT_ENTITY_MATCH == false)
      alternateNames = MentionCompatibility.findPersonAlternateNames(entity.name);
    alternateNames.add(0, entity.name);
    List<List<CoreLabel>> alternateTokens = new ArrayList<List<CoreLabel>>();
    for (String name : alternateNames)
      alternateTokens.add(Utils.tokenize(name));

    // now traverse all sentences that contain this entity (or a mention linked
    // to it in a coreference chain)
    for (CoreMap sentence : sentences) {

      // (A)
      // find all matches of the given entity in this sentence
      // note: this currently uses string matching only,
      // i.e., we do not enforce a specific NE type (PER or ORG) for the
      // matching string
      // we assume that the string is a strong enough disambiguator
      List<EntityMention> entityMentions = extractEntityMatches(alternateNames, alternateTokens, entity.id,
          entity.type, sentence);
      sentence.set(EntityMentionsAnnotation.class, entityMentions);
      // mark these matches so we don't extract them as candidates
      Set<Span> entitySpans = new HashSet<Span>();
      for (EntityMention em : entityMentions)
        entitySpans.add(em.getHead());

      // (B)
      // find additional NEs (i.e., slot candidates) that are conditional on the
      // EntityMentions extracted
      // currently this finds candidates for per:title from the base NPs that
      // include the EntityMention
      findConditionalNamedEntities(entityMentions, sentence);

      // (C)
      // extract slot candidates
      // this simply looks for NEs whose labels match known slot labels
      // for each such NE we create one EntityMention object
      List<EntityMention> slotMentions = extractSlotMentions(sentence, entitySpans);
      sentence.set(SlotMentionsAnnotation.class, slotMentions);

      // (D)
      // create positive/negative relation mentions based on these entities and
      // slots
      // here we check if a slot candidate matches a positive example or not
      List<RelationMention> relations = createPositiveAndNegativeRelations(entityMentions, slotMentions, sentence,
          knownSlots);

      if (relations != null && relations.size() > 0) {

        sentence.set(RelationMentionsAnnotation.class, relations);
        if (stats != null) stats.positiveAndNegativeCounter += relations.size();
        for (RelationMention rel : relations) {
          if (!rel.getType().equals(RelationMention.UNRELATED)) {
            if (stats != null) stats.positiveCounter++;
            myPositiveSlots++;

            logger.severe("FOUND POSITIVE DATUM:\t"
                + rel.getType()
                + "\t"
                + rel.getArg(0).getExtentString()
                + "\t"
                + rel.getArg(1).getExtentString()
                + "\t"
                + (sentence.get(SourceIndexAnnotation.class) != null
                    && sentence.get(SourceIndexAnnotation.class).equals(Constants.WEBINDEX_NAME) ? "true" : "false")
                + "\t'" + Utils.sentenceToMinimalString(sentence).trim() + "'");
          }
        }
      }
      // some stats
      if (stats != null) {
        for (RelationMention rel : relations) {
          if (rel.getType().equals(RelationMention.UNRELATED)) continue;
          String tuple = rel.getArg(0).getExtentString() + ":" + rel.getArg(1).getExtentString();
          String type = rel.getType();
          Set<String> tuples = stats.relationTuples.get(type);
          if (tuples == null) {
            tuples = new HashSet<String>();
            stats.relationTuples.put(type, tuples);
          }
          tuples.add(tuple);
        }
      }

      logger.fine("Sentence after creating relations:\n" + Utils.sentenceToString(sentence));

      // now add all SlotMentionAnnotations to EntityMentionsAnnotation
      // we need this, because this is the way GenericDataSetReader accesses
      // entity mentions
      if (slotMentions != null && slotMentions.size() > 0) {
        List<EntityMention> allEntityMentions = sentence.get(EntityMentionsAnnotation.class);
        assert (allEntityMentions != null);
        allEntityMentions.addAll(slotMentions);
        sentence.set(EntityMentionsAnnotation.class, allEntityMentions);
      }

      // add to corpus if any RelationMentions created
      if (sentence.get(RelationMentionsAnnotation.class) != null) {
        parseSentence(sentence);
        sentencesWithExamples.add(sentence);
      }
    }

    if (stats != null) {
      if (sentences.size() > 0) stats.usefulEntityCount++;
      stats.entityCount++;
      if (stats.entityCount % 100 == 0) {
        logger.severe("Processed " + stats.entityCount + " out of " + entitySlotValues.keySet().size()
            + " entities. Out of these " + stats.usefulEntityCount + " have some relevant sentences.");
      }
    }

    if (runGenericPreprocessor) {
      Annotation corpus = new Annotation("");
      corpus.set(SentencesAnnotation.class, sentencesWithExamples);
      preProcessSentences(corpus);
    }

    logger.severe("ENTITY [" + entity + "]: " + sentencesWithExamples.size() + " relevant sentences with "
        + myPositiveSlots + " slot matches.");
    return sentencesWithExamples;
  }

  private void findConditionalNamedEntities(List<EntityMention> mentions, CoreMap sentence) {
    Tree tree = sentence.get(TreeAnnotation.class);
    if (tree == null) {
      logger
          .severe("WARNING: found sentence without Tree. Will not find conditional NEs, but will continue without...");
      return;
    }

    // make sure the tree contains CoreLabels and tokens are indexed
    convertToCoreLabels(tree);
    if (!((CoreLabel) tree.label()).containsKey(BeginIndexAnnotation.class)) tree.indexSpans(0);

    // find MODIFIERs
    findModifiers(mentions, sentence, tree);
  }

  private void findModifiers(List<EntityMention> mentions, CoreMap sentence, Tree tree) {
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);

    for (EntityMention mention : mentions) {
      Span span = mention.getHead();
      Tree subtree = findTreeHeadedBySpan(tree, span);
      if (subtree == null) continue;
      int start = ((CoreLabel) subtree.label()).get(BeginIndexAnnotation.class);
      int end = span.start();
      assert (start <= end);

      int modifierStart = -1;
      int modifierEnd = -1;
      for (int i = start; i < end; i++) {
        CoreLabel token = tokens.get(i);
        if (modifierStart == -1 && token.tag().startsWith("NN")
            && token.get(NamedEntityTagAnnotation.class).equals("O")) {
          modifierStart = i;
        } else if (modifierStart >= 0
            && (!token.tag().startsWith("NN") || !token.get(NamedEntityTagAnnotation.class).equals("O"))) {
          modifierEnd = i;
        }
      }
      if (modifierStart >= 0) {
        if (modifierEnd == -1) modifierEnd = end;
        StringBuffer os = new StringBuffer();
        for (int i = modifierStart; i < modifierEnd; i++) {
          tokens.get(i).set(NamedEntityTagAnnotation.class, "MODIFIER");
          if (i > modifierStart) os.append(" ");
          os.append(tokens.get(i).word());
        }
        logger.severe("Found modifier [" + os.toString() + "] for entity [" + mention.getExtentString()
            + "] in sentence: " + Utils.sentenceToMinimalString(sentence));
      }
    }
  }

  private Tree findTreeHeadedBySpan(Tree tree, Span span) {
    for (Tree kid : tree.children()) {
      if (kid == null) continue;
      Tree match = findTreeHeadedBySpan(kid, span);
      if (match != null) return match;
    }

    CoreLabel l = (CoreLabel) tree.label();
    if (l != null && l.has(BeginIndexAnnotation.class) && l.has(EndIndexAnnotation.class)) {
      String constLabel = l.value();
      int myStart = l.get(BeginIndexAnnotation.class);
      int myEnd = l.get(EndIndexAnnotation.class);
      if (constLabel.equals("NP") && myStart <= span.start() && myEnd >= span.end()) {
        return tree;
      }
    }

    return null;
  }

  private void onlineNer(List<CoreMap> sentences) {
    correctNerAnnotations(sentences);

    // make sure all AnswerAnnotation slots are set to null (the cached
    // sentences carry this junk around)
    // otherwise, this affects the behavior of the regex NER
    for (CoreMap sentence : sentences) {
      for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
        token.set(AnswerAnnotation.class, null);
      }
    }

    if (regexAnnotator != null) {
      Annotation corpus = new Annotation("");
      corpus.set(SentencesAnnotation.class, sentences);
      regexAnnotator.annotate(corpus);

      // regex eagerly annotates IN/PRP as STATE_OR_PROVINCE, e.g., "Me" or
      // "us", or "as"
      for (CoreMap sentence : sentences) {
        for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
          if ((token.ner().equals("COUNTRY") || token.ner().equals("STATE_OR_PROVINCE"))
              && (token.tag().startsWith("PRP") || token.tag().startsWith("IN"))) {
            token.set(NamedEntityTagAnnotation.class, "O");
          }
        }
      }
    }
  }

  private void correctNerAnnotations(List<CoreMap> sentences) {
    for (CoreMap sentence : sentences) {
      List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);

      //
      // correct mistagging: In "XX year", both tokens get tagged as NUMBER
      // look for 'year' or 'years' that are tagged NUMBER
      //
      for (int i = 0; i < tokens.size(); i++) {
        CoreLabel token = tokens.get(i);
        String nerTag = token.get(NamedEntityTagAnnotation.class);
        if (token.word().startsWith("year") && nerTag.equals("NUMBER")) {
          if (i > 0) { // preceded by a number
            CoreLabel prevToken = tokens.get(i - 1);
            String prevTag = prevToken.get(NamedEntityTagAnnotation.class);

            if (prevTag.equals("NUMBER") && StringUtils.isNumeric(prevToken.word())) {
              token.set(NamedEntityTagAnnotation.class, "O");
            }
          }
        }
      }

      //
      // tag "DD-year-old" as NUMBER
      //
      for (int i = 0; i < tokens.size(); i++) {
        CoreLabel token = tokens.get(i);
        String nerTag = token.get(NamedEntityTagAnnotation.class);
        if (nerTag != null && !nerTag.equals("O")) continue;
        Matcher m = YEAROLD_REGEX.matcher(token.word());
        if (m.matches()) token.set(NamedEntityTagAnnotation.class, "NUMBER");
      }
      //
      // tag "DD-" as NUMBER (apparently "DD-year-old" is tokenized "DD-"
      // "year-old")
      //
      for (int i = 0; i < tokens.size(); i++) {
        CoreLabel token = tokens.get(i);
        String nerTag = token.get(NamedEntityTagAnnotation.class);
        if (nerTag != null && !nerTag.equals("O")) continue;
        Matcher m = DDDASH_REGEX.matcher(token.word());
        if (m.matches()) token.set(NamedEntityTagAnnotation.class, "NUMBER");
      }

      //
      // break date ranges, e.g., "1826 to 1832" into individual dates
      //
      for (int i = 0; i < tokens.size(); i++) {
        if (i < tokens.size() - 2 && tokens.get(i).get(NamedEntityTagAnnotation.class).equals("DATE")
            && tokens.get(i + 1).get(NamedEntityTagAnnotation.class).equals("DATE")
            && tokens.get(i + 2).get(NamedEntityTagAnnotation.class).equals("DATE") && isYear(tokens.get(i).word())
            && isRange(tokens.get(i + 1).word()) && isYear(tokens.get(i + 2).word())) {
          // break year ranges. we care about individual years
          tokens.get(i + 1).set(NamedEntityTagAnnotation.class, "O");
        }
      }
    }
  }

  private static boolean isYear(String s) {
    Matcher m = YEAR_REGEX.matcher(s);
    if (m.matches()) return true;
    return false;
  }

  private static boolean isRange(String s) {
    if (RANGE_WORDS.contains(s.toLowerCase())) return true;
    return false;
  }

  private String setsToString(Set<String> set) {
    List<String> sorted = new ArrayList<String>(set);
    Collections.sort(sorted);
    StringBuffer os = new StringBuffer();
    boolean first = true;
    for (String v : sorted) {
      if (!first) os.append(", ");
      os.append(v);
      first = false;
    }
    return os.toString();
  }

  /**
   * Performs any NLP analysis that was done in the index or cache Includes:
   * tagging of trigger words, syntactic parsing
   * 
   * @param sentence
   */
  private void parseSentence(CoreMap sentence) {
    //
    // Mark relation trigger words
    //
    markTriggerWords(sentence.get(TokensAnnotation.class));

    // make sure sentences are parsed
    Tree tree = sentence.get(TreeAnnotation.class);
    assert (tree != null);
  }

  /**
   * Finds all token spans where this slot matches over the sentence tokens This
   * includes matching the alternates values of the given slot!
   * 
   * @param slot
   * @param sentence
   */
  private void matchSlotInSentence(KBPSlot slot, CoreMap sentence) {
    // this stores all matches in this sentence. must be reset because this is
    // called for multiple sentences
    slot.matchingSpans = new ArrayList<Span>();
    slot.matchingSpanExact = new ArrayList<Boolean>();

    List<String[]> names = new ArrayList<String[]>();
    names.add(slot.slotValueTokens);
    if (slot.alternateSlotValues != null) names.addAll(slot.alternateSlotValues);

    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    boolean[] used = new boolean[tokens.size()];
    Arrays.fill(used, false);
    boolean exact = true;
    for (String[] name : names) {
      for (int start = 0; start < tokens.size();) {
        if (used[start]) { // already taken by another name variant
          start++;
          continue;
        }
        if (nameMatches(name, tokens, start)) {
          logger.fine("MATCHED " + (exact ? "exact" : "alternate") + " slot " + slot.slotName + ":" + slot.slotValue
              + " at position " + start + " in sentence: " + Utils.sentenceToMinimalString(sentence));
          slot.matchingSpans.add(new Span(start, start + name.length));
          slot.matchingSpanExact.add(exact);
          for (int i = 0; i < name.length; i++) {
            used[start + i] = true;
          }
          start += name.length;
        } else {
          start++;
        }
      }
      exact = false;
    }
  }

  private boolean nameMatches(String[] name, List<CoreLabel> tokens, int start) {
    if (start + name.length > tokens.size()) return false;
    for (int i = 0; i < name.length; i++) {
      if (Constants.CASE_INSENSITIVE_SLOT_MATCH) {
        if (!name[i].equalsIgnoreCase(tokens.get(start + i).word())) {
          return false;
        }
      } else {
        if (!name[i].equals(tokens.get(start + i).word())) {
          return false;
        }
      }
    }
    return true;
  }

  @SuppressWarnings("deprecation")
  private List<TemporalWithSpan> getTemporalInformation(CoreMap sentence, AnnotationPipeline pipeline,
      Map<String, String> docDatesMap) throws Exception {

    // there can be multiple temporal expressions in a sentence
    List<TemporalWithSpan> slotsValues = new ArrayList<TemporalWithSpan>();

    int windowMergeTimeExpression = 4;
    try {

      // create document
      Annotation document = new Annotation(Utils.sentenceToMinimalString(sentence));

      pipeline.annotate(document);
      String docId = sentence.get(DocIDAnnotation.class);

      if (docId == null) docId = sentence.get(MachineReadingAnnotations.DocumentIdAnnotation.class);
      if (docId != null) docId = docId.trim();

      document.set(DocIDAnnotation.class, docId);
      document.set(CoreAnnotations.DocIDAnnotation.class, docId);
      String docDate = docDatesMap.get(docId);
      if (docDate != null) docDate = docDate.substring(0, 10);
      document.set(CoreAnnotations.DocDateAnnotation.class, docDate);

      sutime.annotate(document);
      // list of temporal objects that talk about same event, for example
      // "he was in the company from Mar 2003 to May 2007"
      List<List<CoreMap>> timeClusters = new ArrayList<List<CoreMap>>();

      int i = 0;
      for (CoreMap timexAnn : document.get(TimexAnnotations.class)) {

        if (i == 0)
          logger.severe("\n\nsentence is " + Utils.sentenceToMinimalString(sentence) + " and docdate is " + docDate
              + " for doc id " + docId);

        i++;
        TimeExpression tm = timexAnn.get(TimeExpression.Annotation.class);

        edu.stanford.nlp.time.SUTime.Temporal t = tm.getTemporal();
        if (t == null || t.toISOString() == null || t.toISOString().startsWith("P")) {
          logger.fine("ignoring time expression " + t);
          continue;
        }
        int tokenBeginNum = timexAnn.get(CoreAnnotations.TokenBeginAnnotation.class);
        int tokenEndNum = timexAnn.get(CoreAnnotations.TokenEndAnnotation.class);

        boolean foundCluster = false;

        for (List<CoreMap> timeCluster : timeClusters) {
          if (timeCluster.size() == 2) continue;
          for (CoreMap e : timeCluster) {
            int begin = e.get(CoreAnnotations.TokenBeginAnnotation.class);
            int end = e.get(CoreAnnotations.TokenEndAnnotation.class);
            if ((end < tokenBeginNum && tokenBeginNum - end < windowMergeTimeExpression)
                || (tokenEndNum < begin && begin - tokenEndNum < windowMergeTimeExpression)) {
              timeCluster.add(timexAnn);
              foundCluster = true;
              break;
            }
          }
          if (foundCluster) break;
        }
        if (!foundCluster) {
          List<CoreMap> newList = new ArrayList<CoreMap>();
          newList.add(timexAnn);
          timeClusters.add(newList);
        }
        // String dateValue = dateTimex.val();
        // logger.severe("\n\n timex info is " + dateValue + " \n\n");
      }

      for (List<CoreMap> timeCluster : timeClusters) {

        List<TemporalWithSpan> slotV = KBPTemporal.getSlotValues(timeCluster, document
            .get(CoreAnnotations.TokensAnnotation.class));
        if (slotV != null && slotV.size() > 0) slotsValues.addAll(slotV);

        // todo: see NormalizedNamedEntityTagAnnotation

      }

      // when no temporal expressions are present, add temporal values as (null,
      // docDate, docDate, null) if the sentence doesn't contain any past tense
      // verb
      if ((slotsValues == null || slotsValues.size() == 0) && useDocDate) {
        logger.fine("getting temporal information from document date");
        TemporalWithSpan s = KBPTemporal.generateTemporalfrmDocDate(document
            .get(CoreAnnotations.TokensAnnotation.class), docDate);
        if (s != null) slotsValues.add(s);
      }
      logger.fine("slot values are " + StringUtils.join(slotsValues, ";"));
    } catch (Exception e) {
      e.printStackTrace();
    }

    return slotsValues;
  }

  private List<RelationMention> createPositiveAndNegativeRelations(List<EntityMention> entityMentions,
      List<EntityMention> candidateSlotMentions, CoreMap sentence, List<KBPSlot> knownSlots) throws Exception {

    // there can be multiple temporal expressions in a sentence
    List<TemporalWithSpan> slotsValues = null;
    if (temporal) {
      slotsValues = getTemporalInformation(sentence, pipeline, docDatesMap);
    }

    // matches all known slots over tokens in the sentence
    // for now we ignore the candidate NEs, because NEs may match only a subset
    // of the slot value. we handle this later
    for (KBPSlot slot : knownSlots) {

      if (diagnosticMode == true) {
        String sentenceDocId = sentence.get(CoreAnnotations.DocIDAnnotation.class).trim();
        if (!slot.docid.equalsIgnoreCase(sentenceDocId)) {
          logger
              .severe("ignoring sentence since doc id (slot:" + slot.docid + ") doesnt match (" + sentenceDocId + ")");
          continue;
        } else
          logger.severe("Found the match for the document id (slot:" + slot.docid + ") and (" + sentenceDocId + ")");
      }
      logger.fine("Attempting to match slot " + slot.slotName + ":" + slot.slotValue);
      matchSlotInSentence(slot, sentence);
    }

    List<RelationMention> relations = new ArrayList<RelationMention>();
    for (EntityMention slotValue : candidateSlotMentions) {
      // do we match a known slot?
      Map<String, Set<String>> slotTypesAndValues = slotMatchesEntityMentionCandidate(slotValue, knownSlots, sentence);

      for (String normValue : slotTypesAndValues.keySet()) {
        // concatenate all accepted labels for this value, separated by |
        String concatenatedLabel = concatenateLabels(slotTypesAndValues.get(normValue));

        // in diagnostic mode we keep only the positive examples
        // this is needed just for the temporal task, where we extract the
        // timeline of known relations
        if (diagnosticMode && concatenatedLabel.equals(RelationMention.UNRELATED)) {
          continue;
        }

        // create one RelationMention object for each entity mention
        for (EntityMention ent : entityMentions) {
          List<ExtractionObject> args = new ArrayList<ExtractionObject>();
          args.add(ent); // the entity MUST be the first argument (see the
          // dontLexicalizeFirstArg property)
          args.add(slotValue);

          NormalizedRelationMention rm;
          if (temporal) {

            TemporalRelationMention trm = new TemporalRelationMention(normValue, makeRelationMentionId(), sentence,
                ExtractionObject.getSpan(ent, slotValue), concatenatedLabel, null, args);

            if (slotsValues != null && slotsValues.size() > 0) {
              mapTemporalInformation(trm, slotsValues, sentence);
            }
            rm = trm;
          } else {

            rm = new NormalizedRelationMention(normValue, makeRelationMentionId(), sentence, ExtractionObject.getSpan(
                ent, slotValue), concatenatedLabel, null, args, null);

          }
          relations.add(rm);
          relationExampleCount.incrementCount(concatenatedLabel);
          if (!concatenatedLabel.equals(RelationMention.UNRELATED)) {
            logger.fine("Found a positive example for slot type " + concatenatedLabel + ": " + rm);
          }
        }
      }
    }

    return relations;
  }

  private String concatenateLabels(Set<String> labels) {
    StringBuffer os = new StringBuffer();
    boolean first = true;
    List<String> sortedLabels = new ArrayList<String>(labels);
    Collections.sort(sortedLabels);
    for (String label : sortedLabels) {
      if (!first) os.append("|");
      os.append(label);
      first = false;
    }
    String concatenatedLabel = os.toString();
    return concatenatedLabel;
  }

  private void mapTemporalInformation(TemporalRelationMention trm, List<TemporalWithSpan> temporals, CoreMap sentence) {
    boolean verbose = true;

    // keep only temporals with at least one T* slot different from null
    temporals = removeEmptyTemporals(temporals, sentence);
    if (temporals.size() == 0) return;

    // sort in textual order
    Collections.sort(temporals, new Comparator<TemporalWithSpan>() {
      @Override
      public int compare(TemporalWithSpan o1, TemporalWithSpan o2) {
        if (o1.leftMostTokenPosition() < o2.leftMostTokenPosition())
          return -1;
        else if (o1.leftMostTokenPosition() > o2.leftMostTokenPosition())
          return 1;
        else if (o1.rightMostTokenPosition() < o2.rightMostTokenPosition())
          return -1;
        else if (o1.rightMostTokenPosition() == o2.rightMostTokenPosition()) return 0;
        return 1;
      }
    });
    if (verbose && temporals.size() > 1) {
      System.err.println("TEMPORAL MAPPING: found ambiguity; " + temporals.size() + " temporals found: " + temporals);
    }

    // actual mapping
    // pickFirst(trm, temporals, sentence, verbose); // just a baseline
    pickShortestDependency(trm, temporals, sentence, verbose); // mapping based
    // on dependency
    // distance
  }

  private List<TemporalWithSpan> removeEmptyTemporals(List<TemporalWithSpan> temporals, CoreMap sentence) {
    List<TemporalWithSpan> nonEmpties = new ArrayList<TemporalWithSpan>();
    for (TemporalWithSpan t : temporals) {
      if (t.t1 == null && t.t2 == null && t.t3 == null && t.t4 == null) {
        logger
            .severe("TEMPORAL MAPPING: found all-null temporal with span(s) " + t.spans + " in sentence: " + sentence);
      } else {
        nonEmpties.add(t);
      }
    }
    return nonEmpties;
  }

  /**
   * Selects the temporal expression that can be linked to the slot through the
   * shortest dependency For example, in the sentence "The '' ` British 31st
   * Division '' ' was a New Army -LRB- British -RRB- New Army division -LRB-
   * military -RRB- division formed in April 1915 as part of the K4 Army Group
   * and taken over by the War Office on 10 August 1915 .", if the slot is "War
   * Office" it should be linked to "10 August 1915" rather than "April 1915".
   * 
   * @param trm
   * @param temporals
   * @param sentence
   */
  private void pickShortestDependency(TemporalRelationMention trm, List<TemporalWithSpan> temporals, CoreMap sentence,
      boolean verbose) {
    SemanticGraph dependencies = sentence
        .get(SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation.class);
    if (dependencies == null) {
      logger.severe("TEMPORAL MAPPING WARNING: found sentence without dependency annotation: "
          + Utils.sentenceToMinimalString(sentence));
      pickFirst(trm, temporals, sentence, verbose);
      return;
    }
    // System.out.println("semantic graph is " +
    // dependencies.toFormattedString());
    int[][] distanceMatrix = computeDistanceMatrix(dependencies, sentence.get(TokensAnnotation.class).size(), sentence);

    Span slotSpan = trm.getArg(1).getExtent();
    TemporalWithSpan bestTemporal = null;
    int shortestDistance = Integer.MAX_VALUE;
    for (TemporalWithSpan t : temporals) {
      int myDist = computeDependencyDistance(slotSpan, t, distanceMatrix);
      if (myDist < shortestDistance) {
        shortestDistance = myDist;
        bestTemporal = t;
      }
    }

    if (bestTemporal != null) {
      if (bestTemporal != temporals.get(0)) {
        logger.fine("TEMPORAL MAPPING: the dependency picked an element that is not left most: " + bestTemporal
            + " vs " + temporals.get(0));
      }
      mapTemporal(bestTemporal, trm, sentence, verbose);
    } else {
      logger.severe("TEMPORAL MAPPING WARNING: could not select any temporal from this list: " + temporals
          + " in sentence: " + Utils.sentenceToMinimalString(sentence));
      pickFirst(trm, temporals, sentence, verbose);
    }
  }

  /**
   * Computes the dependency distances between all possible token combinations
   * in this sentence
   */
  private int[][] computeDistanceMatrix(SemanticGraph graph, int n, CoreMap sentence) {

    int[][] dist = new int[n][n];
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        dist[i][j] = Integer.MAX_VALUE;

    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    for (int src = 0; src < n; src++) {
      dist[src][src] = 0;
      IndexedWord node0 = graph.getNodeByIndexSafe(src + 1);
      // since these are collapsed dependencies, some nodes are not present in
      // the actual graph
      if (!graph.containsVertex(node0)) continue;
      for (int dst = src + 1; dst < n; dst++) {
        IndexedWord node1 = graph.getNodeByIndexSafe(dst + 1);
        if (!graph.containsVertex(node1)) continue;
        logger.finest("Computing path between nodes " + src + " and " + dst + " in sentence: "
            + Utils.sentenceToMinimalString(sentence));
        List<IndexedWord> path = graph.getShortestUndirectedPathNodes(node0, node1);
        List<SemanticGraphEdge> edgePath = graph.getShortestUndirectedPathEdges(node0, node1);
        if (path != null && edgePath != null) {
          if (!filterDepPaths || validPath(edgePath, node0, node1, tokens) || validPath(edgePath, node1, node0, tokens)) {
            logger.finest("Distance is " + path.size());
            logger.finest("Accepted dependency path between temporal and slot: "
                + FeatureFactory.dependencyPath(edgePath, node0));
            dist[src][dst] = path.size();
            dist[dst][src] = path.size();
          } else {
            logger.finest("Dropped invalid dependency path between temporal and slot: "
                + FeatureFactory.dependencyPath(edgePath, node0));
          }
        }
      }
    }
    return dist;
  }

  private boolean validPath(List<SemanticGraphEdge> edgePath, IndexedWord src, IndexedWord dst, List<CoreLabel> tokens) {
    if (edgePath.size() < 1) return false;

    // a valid path has src the governor of dst
    if (isGovernor(edgePath, src, dst)) {
      return true;
    }

    // a valid path has src directly attached to the verb governing dst
    if (useAttachToGovernorVerb && isAttachedToGovernorVerb(edgePath, src, dst, tokens)) {
      logger.fine("Found attached-to-governor-verb path: " + FeatureFactory.dependencyPath(edgePath, src));
      return true;
    }

    return false;
  }

  private boolean isAttachedToGovernorVerb(List<SemanticGraphEdge> edgePath, IndexedWord src, IndexedWord dst,
      List<CoreLabel> tokens) {
    if (edgePath.size() < 2) return false;

    SemanticGraphEdge firstEdge = edgePath.get(0);
    if (src.equals(firstEdge.getDependent())) {
      IndexedWord top = firstEdge.getGovernor();
      // note: IndexedWord uses indices starting at 1
      CoreLabel topToken = tokens.get(top.index() - 1);
      if (topToken.tag().startsWith("VB") && // the source node is governed by a
          // verb
          isGovernor(edgePath.subList(1, edgePath.size()), top, dst) && // top
          // indeed
          // governs
          // dst
          !dependentsHaveTag("VB", edgePath.subList(1, edgePath.size()), dst, tokens)) { // no
        // other
        // verbs
        // in
        // path
        // from
        // top
        // to
        // dst
        return true;
      }
    }

    return false;
  }

  private boolean dependentsHaveTag(String tag, List<SemanticGraphEdge> edgePath, IndexedWord dst,
      List<CoreLabel> tokens) {
    for (SemanticGraphEdge edge : edgePath) {
      if (dst.equals(edge.getDependent())) {
        // reached the end of the path on the right edges
        // the destination may have any tag (we are not checking dst)
        break;
      } else {
        IndexedWord dep = edge.getDependent();
        CoreLabel depToken = tokens.get(dep.index() - 1);
        if (depToken.tag().startsWith(tag)) {
          return true;
        }
      }
    }
    return false;
  }

  private boolean isGovernor(List<SemanticGraphEdge> edgePath, IndexedWord src, IndexedWord dst) {
    for (SemanticGraphEdge edge : edgePath) {
      if (src.equals(edge.getGovernor())) {
        if (dst.equals(edge.getDependent())) {
          // reached the end of the path on the right edges
          return true;
        } else {
          // so far so good; moving on
          src = edge.getDependent();
        }
      } else {
        // found an edge where src is not governor; stop
        break;
      }
    }
    return false;
  }

  private int computeDependencyDistance(Span slotSpan, TemporalWithSpan temporal, int[][] distanceMatrix) {
    // System.out.println("temporal spans is " + temporal.spans != null ?
    // StringUtils.join(temporal.spans, ";") : null);
    if (temporal.spans == null) return Integer.MAX_VALUE;
    int smallestDistance = Integer.MAX_VALUE;
    try {
      for (Span span : temporal.spans) {
        for (int src = span.start(); src < span.end(); src++) {
          // the temporal expression cannot be part of the slot
          if (src >= slotSpan.start() && src < slotSpan.end()) continue;
          for (int dst = slotSpan.start(); dst < slotSpan.end(); dst++) {
            if (distanceMatrix[src][dst] < smallestDistance) {
              smallestDistance = distanceMatrix[src][dst];
            }
          }
        }
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
    return smallestDistance;
  }

  /**
   * Selects the temporal expression that appears first (left most) in the
   * sentence This is just a baseline
   * 
   * @param trm
   * @param temporals
   */
  private void pickFirst(TemporalRelationMention trm, List<TemporalWithSpan> temporals, CoreMap sentence,
      boolean verbose) {
    mapTemporal(temporals.get(0), trm, sentence, verbose);
  }

  private void mapTemporal(TemporalWithSpan chosen, TemporalRelationMention trm, CoreMap sentence, boolean verbose) {
    trm.t1Slot = chosen.t1;
    trm.t2Slot = chosen.t2;
    trm.t3Slot = chosen.t3;
    trm.t4Slot = chosen.t4;

    if (verbose) {
      logger.severe("TEMPORAL MAPPING: mapped temporal " + chosen + " to relation mention: " + trm + " in sentence: "
          + Utils.sentenceToMinimalString(sentence));
    }
  }

  /**
   * Returns the <slot type, slot value> for all slots that are compatible with
   * this candidate
   * 
   * @param candidate
   * @param knownSlots
   * @param sentence
   * @return A map from normalized slot value to set of labels matched for this
   *         value
   */
  private Map<String, Set<String>> slotMatchesEntityMentionCandidate(EntityMention candidate,
      Collection<KBPSlot> knownSlots, CoreMap sentence) {
    Map<String, Set<String>> matchingTypes = new HashMap<String, Set<String>>();

    if (testMode && diagnosticMode == false) {
      // testMode: testing over some queries in KBPTester => create all examples
      // mark as unrelated (will be changed by the classifier)
      matchingTypes.put(candidate.getExtentString(), new HashSet<String>(Arrays.asList(RelationMention.UNRELATED)));
      return matchingTypes;
    }

    // mode: running train/test over the KB entities
    for (KBPSlot slot : knownSlots) {
      String slotType = slot.slotName;
      Set<String> validNEs = null;
      if (matchSlotNE) {
        // System.err.println("slotsToNamedEntities contains info on the following slots: "
        // + slotsToNamedEntities.keySet());
        // System.err.println("slotType is " + slotType);

        validNEs = slotsToNamedEntities.getSlotInfo(slotType).validNamedEntityLabels();
        if (validNEs == null) throw new RuntimeException("ERROR: cannot find valid NEs for relation " + slotType);
      }
      if (!matchSlotNE || !enforceNeInTraining || validNEs.contains(candidate.getType())) {
        if (Constants.EXACT_SLOT_MATCH) {
          if (slotMatchesExactly(candidate, slot, sentence)) {
            addType(matchingTypes, slot.slotValue, slotType);
            if (!allowMultipleSlotMatchesPerMention) break;
          }
        } else {
          if (slotMatchesExactOrAlternates(candidate, slot, sentence)) {
            addType(matchingTypes, slot.slotValue, slotType);
            if (!allowMultipleSlotMatchesPerMention) break;
          }
        }
      }
    }
    // found at least one positive match
    if (matchingTypes.size() > 0) return matchingTypes;

    // this candidate does not match any known slots
    matchingTypes.put(candidate.getExtentString(), new HashSet<String>(Arrays.asList(RelationMention.UNRELATED)));
    return matchingTypes;
  }

  private static void addType(Map<String, Set<String>> matchingTypes, String value, String type) {
    Set<String> types = matchingTypes.get(value);
    if (types == null) {
      types = new HashSet<String>();
      matchingTypes.put(value, types);
    }
    types.add(type);
  }

  /**
   * Finds if the candidates matches either the slot value or any of alternates
   * that we built in MentionCompatibility.findAlternateSlotValues()
   * 
   * @param candidate
   * @param slot
   * @param sentence
   */
  private boolean slotMatchesExactOrAlternates(EntityMention candidate, KBPSlot slot, CoreMap sentence) {

    // this can happen during diagnostic test
    if (slot.matchingSpans == null || slot.matchingSpans.size() == 0) {
      logger.fine("slot matching spans is empty.");
      return false;
    }
    // matchingSpans show where either the exact slot value or
    // an alternate value matched in the sentence
    for (int i = 0; i < slot.matchingSpans.size(); i++) {
      Span slotSpan = slot.matchingSpans.get(i);
      boolean isExact = slot.matchingSpanExact.get(i);

      //
      // exact match between candidate NE and the slot or one of its alternate
      // values
      //
      if (slotSpan.equals(candidate.getHead())) {
        logger.fine("EXACT MATCH between NE \"" + candidate.getExtentString() + "\" and slot: " + slot.slotName + ":"
            + slot.slotValue + " in sentence: " + Utils.sentenceToMinimalString(sentence));
        return true;
      }

      //
      // entity fully included in slot
      // this may happen, our NER is more conservative than KBP annotations
      // e.g., "Westside College" is included in "Westside College, London"
      // note: this is only relevant during training
      //
      if (candidate.getHeadTokenStart() >= slotSpan.start() && candidate.getHeadTokenEnd() <= slotSpan.end()) {
        logger.fine("INCLUSION MATCH between NE \"" + candidate.getExtentString() + "\" and slot: " + slot.slotName
            + ":" + slot.slotValue + " in sentence: " + Utils.sentenceToMinimalString(sentence));
        return true;
      }

      //
      // slot fully included in entity
      // allow this *only* for date slots and exact matches,
      // e.g., NE:May 1998 slot: 1998
      // for all others, this does not hold!
      // TODO: this may no longer be needed because the alternate slot values
      // include just the year for date slots.
      //
      if (isExact && KBPSlot.isDateSlot(slot.slotName) && candidate.getHeadTokenStart() <= slotSpan.start()
          && candidate.getHeadTokenEnd() >= slotSpan.end()) {
        logger.fine("CONTAINMENT MATCH between NE \"" + candidate.getExtentString() + "\" and slot: " + slot.slotName
            + ":" + slot.slotValue + " in sentence: " + Utils.sentenceToMinimalString(sentence));
        return true;
      }
    }

    return false;
  }

  private boolean slotMatchesExactly(EntityMention candidate, KBPSlot slot, CoreMap sentence) {
    String regex = entityMentionToRegex(candidate, sentence);
    Pattern p = Pattern.compile(regex, (Constants.CASE_INSENSITIVE_SLOT_MATCH ? Pattern.CASE_INSENSITIVE : 0));
    Matcher m = p.matcher(slot.slotValue);
    if (m.matches()) return true; // full match between NE and slot value
    return false;
  }

  private String entityMentionToRegex(EntityMention candidate, CoreMap sentence) {
    StringBuffer os = new StringBuffer();
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    for (int i = candidate.getHeadTokenStart(); i < candidate.getHeadTokenEnd(); i++) {
      if (i > candidate.getHeadTokenStart()) {
        os.append("\\s+");
      }
      os.append(Utils.escapeSpecialRegexCharacters(tokens.get(i).word()));
    }
    return os.toString();
  }

  private List<EntityMention> extractSlotMentions(CoreMap sentence, final Set<Span> entitySpans) {
    List<EntityMention> slots = new ArrayList<EntityMention>();
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    for (int start = 0; start < tokens.size();) {
      String ner = tokens.get(start).ner();
      String pos = tokens.get(start).tag();
      StringBuffer value = new StringBuffer();
      value.append(tokens.get(start).get(TextAnnotation.class));

      // valid candidates must be NEs
      if (ner == null || ner.equals("O")) {
        start++;
        continue;
      }

      // System.err.println("start = " + start + " " + ner + "/" + pos + "/" +
      // tokens.get(start).word());
      int end = start + 1;
      while (end < tokens.size()) {
        CoreLabel crt = tokens.get(end);
        // System.err.println("end = " + end + " " + crt.ner() + "/" + crt.tag()
        // + "/" + crt.word());
        if (crt.ner() == null || !crt.ner().equals(ner)) {
          break;
        }

        value.append(" ");
        value.append(crt.get(TextAnnotation.class));
        end++;
      }

      // if not valid, move on
      if (SlotValidity.validCandidate(slotsToNamedEntities, value.toString(), ner, pos, matchSlotNE)) {
        Span span = new Span(start, end);
        if (!overlaps(span, entitySpans) && closeEnough(span, entitySpans)) {
          // String text = Utils.sentenceSpanString(tokens, span);
          // if(end - start > 1) System.err.println("FOUND MULTICANDIDATE:");
          // System.err.println("FOUND CANDIDATE: [" + text + "]/" + ner +
          // " in sentence: " + Utils.sentenceToString(sentence));
          EntityMention em = new EntityMention(makeEntityMentionId(null), sentence, span, span, ner, null, null);
          slots.add(em);
        }
      }

      start = end;
    }
    return slots;
  }

  private boolean closeEnough(Span slotSpan, Set<Span> entitySpans) {
    for (Span entitySpan : entitySpans) {
      if (slotSpan.end() <= entitySpan.start()
          && entitySpan.start() - slotSpan.end() < Constants.MAX_DISTANCE_BETWEEN_ENTITY_AND_SLOT) {
        return true;
      } else if (entitySpan.end() <= slotSpan.start()
          && slotSpan.start() - entitySpan.end() < Constants.MAX_DISTANCE_BETWEEN_ENTITY_AND_SLOT) {
        return true;
      }
    }
    return false;
  }

  private boolean overlaps(Span span, Set<Span> allSpans) {
    if (allSpans.contains(span)) return true;
    for (Span otherSpan : allSpans) {
      if (span.start() < otherSpan.start() && span.end() > otherSpan.start()) {
        return true;
      }
      if (span.start() >= otherSpan.start() && span.start() < otherSpan.end()) {
        return true;
      }
    }
    return false;
  }

  /**
   * Finds all matches of the given KBP entity in this sentence
   * 
   * @return A list of machinereading.EntityMention objects, one for each
   *         sentence match
   */
  private List<edu.stanford.nlp.ie.machinereading.structure.EntityMention> extractEntityMatches(
      List<String> entityNames, List<List<CoreLabel>> entitiesTokens, String entityId, EntityType type, CoreMap sentence) {
    List<EntityMention> entities = new ArrayList<EntityMention>();
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    assert (tokens != null);

    // traverse all valid forms of this entity name
    assert (entityNames.size() == entitiesTokens.size());
    for (int idx = 0; idx < entityNames.size(); idx++) {
      String entityName = entityNames.get(idx);
      List<CoreLabel> entityTokens = entitiesTokens.get(idx);

      //
      // 1. look for exact matches of entityTokens (which are just the tokens in
      // entityName) over sentence tokens
      // 2. if no exact match check if the antecedent of a token is the actual
      // entity we care about
      //
      boolean[] used = new boolean[tokens.size()]; // true if this token already
      // used for a mention
      Arrays.fill(used, false);
      for (int start = 0; start < tokens.size(); start++) {
        if (used[start]) continue;
        boolean failed = start > tokens.size() - entityTokens.size();
        if (!failed) {
          for (int i = 0; i < entityTokens.size() && !failed; ++i) {
            String needle = entityTokens.get(i).word();
            String haystack = tokens.get(start + i).word();
            if (Constants.CASE_INSENSITIVE_ENTITY_MATCH == false && !needle.equals(haystack)) {
              failed = true;
              break;
            }
            if (Constants.CASE_INSENSITIVE_ENTITY_MATCH && !needle.equalsIgnoreCase(haystack)) {
              failed = true;
              break;
            }
          }
        }
        if (!failed) {
          Span span = new Span(start, start + entityTokens.size());
          EntityMention em = new EntityMention(makeEntityMentionId(entityId), sentence, span, span, Utils
              .makeEntityType(type), null, null);
          entities.add(em);
          for (int i = span.start(); i < span.end(); i++)
            used[i] = true;
          logger
              .fine("Found match for name #" + idx + " " + em + " at position " + start + " in sentence: " + sentence);
          if (idx > 0) logger.fine("Full name was: " + entityNames.get(0));
        } else {
          // checking using coref info
          if (Constants.USE_COREF) {
            String antecedent = tokens.get(start).get(CoreAnnotations.AntecedentAnnotation.class);
            if (antecedent != null
                && ((Constants.CASE_INSENSITIVE_ENTITY_MATCH == false && antecedent.equals(entityName)) || (Constants.CASE_INSENSITIVE_ENTITY_MATCH == true && antecedent
                    .equalsIgnoreCase(entityName))) && validMentionForKBP(tokens.get(start), antecedent)) {
              Span span = new Span(start, start + 1);
              EntityMention em = new EntityMention(makeEntityMentionId(entityId), sentence, span, span, Utils
                  .makeEntityType(type), null, null);
              logger.fine("Found coref entity: " + tokens.get(start) + " that has entity antecedent \"" + antecedent
                  + "\" in sentence: " + Utils.sentenceToMinimalString(sentence));
              entities.add(em);
              for (int i = span.start(); i < span.end(); i++)
                used[i] = true;

              // if coref matched, we want to fake the real entity as much as
              // possible
              // since in training we train only over names, change its POS to
              // NNP if not NNP already
              if (!tokens.get(start).tag().startsWith("NNP")) {
                tokens.get(start).setTag("NNP");
              }
            }
          }
        }
      }
    }
    return entities;
  }

  private boolean validMentionForKBP(CoreLabel mention, String antecedent) {
    // accept pronouns
    if (mention.tag().startsWith("PRP")) {
      return true;
    }

    // accept nouns only if included in antecedent
    // why: coref is still weird for nominal coreference...
    if (mention.tag().startsWith("NN") && antecedent.toLowerCase().contains(mention.word().toLowerCase())) {
      return true;
    }

    return false;
  }

  private String makeEntityMentionId(String kbpId) {
    String id = "EM" + entityMentionCount;
    if (kbpId != null) id += "-KBP" + kbpId;
    entityMentionCount++;
    return id;
  }

  public static String extractEntityId(String id) {
    int start = id.indexOf("-KBP");
    if (start < 0 || start >= id.length() - 4)
      throw new RuntimeException("Found entity mention with invalid id: " + id);
    return id.substring(start + 4);
  }

  private String makeRelationMentionId() {
    String id = "RM" + relationMentionCount;
    relationMentionCount++;
    return id;
  }

  private void markTriggerWords(List<CoreLabel> tokens) {
    if (triggers == null) return;

    for (TriggerSeq seq : triggers) {
      for (int start = 0; start < tokens.size() - seq.tokens.length;) {

        if (matches(tokens, seq.tokens, start)) {
          tokens.get(start).set(TriggerAnnotation.class, "B-" + seq.label);
          for (int i = 1; i < seq.tokens.length; i++)
            tokens.get(start + i).set(TriggerAnnotation.class, "I-" + seq.label);

          start += seq.tokens.length;
        } else {
          start++;
        }
      }
    }
  }

  private boolean matches(List<CoreLabel> tokens, String[] triggerTokens, int start) {
    for (int i = 0; i < triggerTokens.length; i++) {
      if (!tokens.get(start + i).word().equalsIgnoreCase(triggerTokens[i])) return false;
    }
    logger.fine("Matched trigger sequence " + triggerTokens.toString() + " at position " + start + " in sentence "
        + StringUtils.join(tokens));
    return true;
  }

  private void loadTriggerWords(String fn) throws IOException {
    BufferedReader is = new BufferedReader(new FileReader(fn));
    triggers = new ArrayList<TriggerSeq>();
    String line;
    while ((line = is.readLine()) != null) {
      line = line.trim();
      int firstTab = line.indexOf('\t');
      if (firstTab < 0) {
        firstTab = line.indexOf(' ');
      }
      assert (firstTab > 0 && firstTab < line.length());
      String label = line.substring(0, firstTab).trim();
      List<CoreLabel> tokens = Utils.tokenize(line.substring(firstTab).trim());
      String[] words = new String[tokens.size()];
      for (int i = 0; i < tokens.size(); i++)
        words[i] = tokens.get(i).word();
      triggers.add(new TriggerSeq(label, words));
    }
    is.close();

    // make sure trigger sequences are sorted in descending order of length so
    // we always match the longest sequence first
    Collections.sort(triggers);
    logger.info("Loaded " + triggers.size() + " trigger sequences.");
  }

  private static File[] fetchFiles(File dir) {
    if (dir.isDirectory())
      return dir.listFiles();
    else {
      File[] files = new File[1];
      files[0] = dir;
      return files;
    }
  }

  private static String normalizeEntityType(String s) {
    if (s.contains("PER")) return "PER";
    if (s.contains("ORG")) return "ORG";
    throw new RuntimeException("Unknown entity type: " + s);
  }

  private List<DatumAndMention> annotationToDatums(Annotation corpus, FeatureFactory rff, Counter<String> domainStats) {
    List<DatumAndMention> datums = new ArrayList<DatumAndMention>();

    for (CoreMap sentence : corpus.get(SentencesAnnotation.class)) {
      List<RelationMention> relationMentions = sentence.get(MachineReadingAnnotations.RelationMentionsAnnotation.class);
      if (relationMentions != null) {
        for (RelationMention rel : relationMentions) {
          Datum<String, String> d = rff.createDatum(rel);

          if (domainAdaptation) {
            CoreMap sent = rel.getSentence();
            if (sent == null) {
              // should not happen
              throw new RuntimeException("ERROR: failed to find sentence for relation mention: " + rel);
            }
            String indexPath = Constants.getIndexPath(sent.get(SourceIndexAnnotation.class));
            String domainName = Constants.indexToDomain(indexPath, domainAdaptationStyle);
            domainStats.incrementCount(domainName);
            d = applyDaumeDomainAdaptation(d, domainName);
          }

          assert (rel instanceof NormalizedRelationMention);
          datums.add(new DatumAndMention(d, (NormalizedRelationMention) rel));
        }
      }
    }

    return datums;
  }

  private static Datum<String, String> applyDaumeDomainAdaptation(Datum<String, String> original, String domainName) {
    assert (original instanceof BasicDatum<?, ?>);
    Collection<String> origFeats = original.asFeatures();
    Collection<String> adaptedFeats = new ArrayList<String>();
    for (String origFeat : origFeats) {
      adaptedFeats.add(origFeat);
      adaptedFeats.add(domainName + ">" + origFeat);
    }
    return new BasicDatum<String, String>(adaptedFeats, original.label());
  }

  public final List<DatumAndMention> generateDatums(Annotation corpus, FeatureFactory rff, Counter<String> labelStats,
      Counter<String> domainStats) {

    // create the datums for all mentions
    List<DatumAndMention> datums = annotationToDatums(corpus, rff, domainStats);
    logger.severe("Finished converting shard to datums. Constructed " + datums.size() + " datums.");

    // build stats
    if (labelStats != null) {
      for (DatumAndMention dm : datums) {
        labelStats.incrementCount(dm.datum().label());
      }
    }

    return datums;
  }

  public final void parse(PrintStream os, String path, FeatureFactory rff, Counter<String> labelStats,
      Counter<String> domainStats) throws Exception {
    File[] files = fetchFiles(new File(path));

    entToSentHistogram = new ClassicCounter<Integer>();
    relationExampleCount = new ClassicCounter<String>();
    ReadStats stats = new ReadStats();

    // traverse each KB file (although we are commonly called with just one
    // file)
    for (File file : files) {
      //
      // read (entity -- slot value -- slot name) tuples from either the KB
      // (training) or the query file (testing)
      // note: during testing, slot value and slot name are not populated
      //
      Map<KBPEntity, List<KBPSlot>> entitySlotValues = loadEntitiesAndSlots(file.getAbsolutePath());

      // sort entities alphabetically
      List<KBPEntity> sortedEntities = new ArrayList<KBPEntity>(entitySlotValues.keySet());
      Collections.sort(sortedEntities, new Comparator<KBPEntity>() {
        public int compare(KBPEntity o1, KBPEntity o2) {
          return o1.name.compareTo(o2.name);
        }
      });

      // extract sentences for each individual entity
      for (KBPEntity entity : sortedEntities) {
        // all calls to readEntity are expensive, so let's do
        // one entity at a time to save memory
        List<CoreMap> sentences = readEntity(entity, entitySlotValues, file, stats, true);
        Annotation corpus = new Annotation("");
        corpus.set(SentencesAnnotation.class, sentences);

        // create datums
        // optionally, merge RelationMention objects with the same slot into a
        // single datum
        List<DatumAndMention> dms = generateDatums(corpus, rff, labelStats, domainStats);

        // save the datum object, discard RelationMention objects
        List<MinimalDatum> datumsOutput = new ArrayList<MinimalDatum>();
        for (DatumAndMention dm : dms) {
          datumsOutput.add(new MinimalDatum(dm.mention().getArg(0).getObjectId(), normalizeEntityType(dm.mention()
              .getArg(0).getType()), dm.mention().getArg(1).getType(), dm.mention().getNormalizedSlotValue(), dm
              .datum()));
        }

        // save datums to stream
        for (MinimalDatum datum : datumsOutput) {
          if (!(datum.datum() instanceof BasicDatum<?, ?>)) {
            throw new RuntimeException("Datums must be BasicDatums here! This should NOT happen...");
          }
          datum.saveDatum(os);
        }
      }

      logger.severe("Finished parsing shard: " + file.getAbsolutePath());
    }
  }

  public static class ReadStats {
    Map<String, Set<String>> relationTuples = new HashMap<String, Set<String>>();
    int positiveAndNegativeCounter = 0;
    int positiveCounter = 0;
    int sentenceCounter = 0;
    int slotCount = 0;
    int entityCount = 0;
    int usefulEntityCount = 0;
  };

  static class TriggerSeq implements Comparable<TriggerSeq> {
    String label;
    String[] tokens;

    public String toString() {
      StringBuffer os = new StringBuffer();
      os.append(label + ":");
      for (String t : tokens)
        os.append(" " + t);
      return os.toString();
    }

    public TriggerSeq(String l, String[] ts) {
      label = l;
      tokens = ts;
    }

    /** Descending order of lengths */
    public int compareTo(TriggerSeq other) {
      return other.tokens.length - this.tokens.length;
    }
  };
}