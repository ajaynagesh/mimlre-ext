package edu.stanford.nlp.kbp.slotfilling;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
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

import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

import com.sri.faust.gazetteer.Gazetteer;
import com.sri.faust.gazetteer.maxmind.MaxmindGazetteer;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationExtractorFactory;
import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.kbp.slotfilling.common.DatumAndMention;
import edu.stanford.nlp.kbp.slotfilling.common.EntityType;
import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.KBPSlot;
import edu.stanford.nlp.kbp.slotfilling.common.KBPTuple;
import edu.stanford.nlp.kbp.slotfilling.common.ListOutput;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Nationalities;
import edu.stanford.nlp.kbp.slotfilling.common.NormalizedRelationMention;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.kbp.slotfilling.common.SlotToNamedEntities;
import edu.stanford.nlp.kbp.slotfilling.common.SlotType;
import edu.stanford.nlp.kbp.slotfilling.common.SlotsToNamedEntities;
import edu.stanford.nlp.kbp.slotfilling.common.Utils;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.KBPReader;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.TaskXMLParser;
import edu.stanford.nlp.kbp.slotfilling.index.DocidFinder;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.ThreeDimensionalMap;

public class KBPEvaluator {
  /** Working directory */
  private final String workDir;

  /** Parses KBP related corpora */
  private final KBPReader reader;

  /** These are all the known entries in the full KBP KB */
  private final Map<KBPEntity, List<KBPSlot>> kbEntities;

  /** The actual relation classifier */
  private final RelationExtractor relationExtractor;

  /** The feature factory used by relationExtractor */
  private final FeatureFactory rff;

  /**
   * Mapping from slot types to other information about them (single v. list,
   * expected NER type)
   */
  private final SlotsToNamedEntities slotsToNamedEntities;

  /**
   * The path of the index where we need to find docs. Other docs are nice, but
   * only ones from this index can be correct results
   */
  private final String officialIndex;
  private final DocidFinder docidFinder;

  /**
   * Map from nationalities to country names The keys are nationalities in lower
   * case, values are case-sensitive country names
   */
  private Nationalities nationalities;

  /** Needed for location inference in KBPInference */
  private final Gazetteer gazetteer;
  private final Map<String, Set<String>> stateAbbreviations;
  private final Map<String, String> stateAbbreviationsToFullName;
  
  /** Decides if two mentions should be clustered together in the same relation */
  private final MentionCompatibility mentionCompatibility;

  /** Do location and temporal inference */
  private final boolean doDomainSpecInference;

  private static boolean temporal;

  private static boolean diagnosticMode = false;

  /** Contains pairs of slot names that are allowed to overlap on the same slot */
  private final Set<String> overlappingRelations;

  /** Overall domain stats (if domain adaptation is used) */
  private Counter<String> domainStats;

  public KBPEvaluator(Properties props) throws Exception {
    String logLevel = props.getProperty(Props.LOG_LEVEL, "INFO");
    Log.setLevel(Log.stringToLevel(logLevel));

    temporal = PropertiesUtils.getBool(props, Props.KBP_TEMPORAL, false);
    diagnosticMode = PropertiesUtils.getBool(props, Props.KBP_DIAGNOSTICMODE, false);

    reader = new KBPReader(props, true, temporal, diagnosticMode);
    reader.setLoggerLevel(Log.stringToLevel(props.getProperty(Props.READER_LOG_LEVEL, "INFO")));

    // load map from nationalities to countries
    if (props.containsKey(Props.NATIONALITIES))
      nationalities = new Nationalities(props.getProperty(Props.NATIONALITIES));
    stateAbbreviations = new HashMap<String, Set<String>>();
    stateAbbreviationsToFullName = new HashMap<String, String>();
    loadStateAbbreviations(props.getProperty(Props.STATES, "/u/nlp/data/TAC-KBP2010/states/state-abbreviations.txt"),
        stateAbbreviations, stateAbbreviationsToFullName);
    gazetteer = new MaxmindGazetteer();

    kbEntities = reader.parseKnowledgeBase(props.getProperty(Props.INPUT_KB));

    slotsToNamedEntities = new SlotsToNamedEntities(props.getProperty(Props.NERENTRY_FILE));

    // TODO: fix absolute path
    overlappingRelations = loadOverlappingRelations(props.getProperty(Props.OVERLAPPING_RELATIONS,
        "/u/nlp/data/TAC-KBP2010/overlaps.tab"), slotsToNamedEntities);

    Log.info("Loading relation extractor...");
    workDir = props.getProperty(Props.WORK_DIR);
    RelationExtractorFactory factory = 
      new RelationExtractorFactory(props.getProperty(Props.MODEL_TYPE, Constants.DEFAULT_MODEL));
    double samplingRatio = PropertiesUtils.getDouble(props, Props.NEGATIVES_SAMPLE_RATIO,
        Constants.DEFAULT_NEGATIVES_SAMPLING_RATIO);
    String srn = props.getProperty(Props.SERIALIZED_MODEL_PATH, "kbp_relation_model");
    if (srn.endsWith(Constants.SER_EXT)) srn = srn.substring(0, srn.length() - Constants.SER_EXT.length());
    String modelPath = KBPTrainer.makeModelPath(workDir, srn, factory.modelType(), samplingRatio);

    relationExtractor = factory.load(modelPath, props);
    String[] relationFeatures = props.getProperty(Props.RELATION_FEATS).split(",\\s*");
    assert (relationFeatures != null && relationFeatures.length > 0);
    Log.severe("relationFeatures: " + StringUtils.join(relationFeatures));
    rff = new FeatureFactory(relationFeatures);
    rff.setDoNotLexicalizeFirstArgument(true);

    officialIndex = props.getProperty(DocidFinder.OFFICIAL_INDEX_PROPERTY);

    // if the datum cache for the test queries exists, we do not need the docidFinder
    // the information provided by docidFinder is already cached
    File testCacheDir = new File(workDir + File.separator + "test");
    File [] caches = null;
    if(testCacheDir.exists() &&
            testCacheDir.isDirectory() &&
            (caches = testCacheDir.listFiles()) != null &&
            caches.length > 0) {
      docidFinder = null;
    } else {
      docidFinder = new DocidFinder(officialIndex);
    }

    mentionCompatibility = new MentionCompatibility(nationalities, slotsToNamedEntities);
    domainStats = new ClassicCounter<String>();

    doDomainSpecInference = PropertiesUtils.getBool(props, Props.DOMAIN_SPEC_INFERENCE, false);
  }

  private static void loadStateAbbreviations(String fn, Map<String, Set<String>> abbrevs,
      Map<String, String> abbrevsToFullName) throws IOException {
    BufferedReader is = new BufferedReader(new FileReader(fn));
    for (String line; (line = is.readLine()) != null;) {
      line = line.toLowerCase();
      String[] bits = line.split("\t");
      assert (bits.length > 1);
      String state = bits[0];
      Set<String> sa = new HashSet<String>();
      for (int i = 1; i < bits.length; i++) {
        sa.add(bits[i]);
        abbrevsToFullName.put(bits[i], state);
        if (bits[i].endsWith(".")) {
          String s = bits[i].substring(0, bits[i].length() - 1);
          abbrevsToFullName.put(s, state);
        }
      }
      abbrevs.put(state, sa);
    }
    is.close();
  }

  private static Set<String> loadOverlappingRelations(String fn, SlotsToNamedEntities slotsToNamedEntities)
      throws IOException {
    Set<String> overlapping = new HashSet<String>();
    BufferedReader is = new BufferedReader(new FileReader(fn));
    for (String line; (line = is.readLine()) != null;) {
      if (line.startsWith("#")) continue;
      String[] bits = line.split("\t");
      if (bits.length != 2) throw new RuntimeException("ERROR: invalid line in overlapping relations file: " + line);
      String n1 = bits[0];
      SlotToNamedEntities s1 = slotsToNamedEntities.getSlotInfo(n1);
      assert (s1 != null);
      String n2 = bits[1];
      SlotToNamedEntities s2 = slotsToNamedEntities.getSlotInfo(n2);
      assert (s2 != null);
      if (intersection(s1.validNamedEntityLabels(), s2.validNamedEntityLabels()).size() > 0) {
        String v = (n1.compareTo(n2) < 0 ? n1 + "\t" + n2 : n2 + "\t" + n1);
        overlapping.add(v);
      }

    }

    is.close();
    Log.severe("Found " + overlapping.size() + " overlapping relations: " + overlapping);
    return overlapping;

  }

  private static <E> Collection<E> intersection(Collection<E> s1, Collection<E> s2) {
    Collection<E> inters = new HashSet<E>();
    for (E e1 : s1) {
      if (s2.contains(e1)) {
        inters.add(e1);
      }
    }
    return inters;
  }

  private boolean overlappingSlot(String s1, String s2) {
    String v = (s1.compareTo(s2) < 0 ? s1 + "\t" + s2 : s2 + "\t" + s1);
    return overlappingRelations.contains(v);
  }

  private SlotsToNamedEntities getSlotsToNamedEntities() {
    return slotsToNamedEntities;
  }

  static void cleanOutputFile(String goldFile, String outputFile, String runId) throws IOException {
    Map<String, ThreeDimensionalMap<String, String, String, Boolean>> map = new HashMap<String, ThreeDimensionalMap<String, String, String, Boolean>>();

    BufferedWriter wNewFile = new BufferedWriter(new FileWriter(outputFile + "_cleaned"));

    for (String line : IOUtils.readLines(goldFile)) {
      String[] tokens = line.split("\\s+", 5);
      assert (tokens.length == 5);
      String queryId = tokens[0].trim();
      String slotName = tokens[1].trim().replaceAll("/", "SLASH");
      String docId = tokens[3].trim();
      String slotValue = tokens[4].trim();
      if (!map.containsKey(queryId)) map.put(queryId, new ThreeDimensionalMap<String, String, String, Boolean>());
      map.get(queryId).put(slotName, slotValue, docId, false);
    }

    for (String line : IOUtils.readLines(outputFile)) {
      String[] tokens = line.split("\\s+", 7);
      assert (tokens.length == 7);
      String queryId = tokens[0].trim();
      String slotName = tokens[1].trim().replaceAll("/", "SLASH");
      String docId = tokens[5].trim();
      String slotValue = tokens[6].trim();

      if (!map.containsKey(queryId) || !map.get(queryId).contains(slotName, slotValue, docId)) continue;
      wNewFile.write(line + "\n");
      map.get(queryId).put(slotName, slotValue, docId, true);
    }

    for (Map.Entry<String, ThreeDimensionalMap<String, String, String, Boolean>> en : map.entrySet()) {
      String queryId = en.getKey();
      for (String slotName : en.getValue().firstKeySet()) {
        for (String slotValue : en.getValue().get(slotName).firstKeySet()) {
          for (String docId : en.getValue().get(slotName).get(slotValue).keySet()) {
            if (en.getValue().get(slotName, slotValue, docId) == false) {
              wNewFile.write(queryId + "\t" + slotName + "\t" + "V" + "\t-\t" + runId + "\t" + docId + "\t" + slotValue
                  + "\n");

            }
          }
        }
      }
    }
    wNewFile.close();
  }
  
  static void outputRelations(
      PrintStream os, Properties props, 
      SlotsToNamedEntities nerTypes,
      Map<KBPEntity, Collection<KBPSlot>> rawRelations, 
      boolean modelCombinationMode) {
    outputRelations(os, props, nerTypes, rawRelations, modelCombinationMode, 0.0);
  }

  static void outputRelations(
      PrintStream os, Properties props, 
      SlotsToNamedEntities nerTypes,
      Map<KBPEntity, Collection<KBPSlot>> rawRelations, 
      boolean modelCombinationMode,
      double threshold) {
    String runId = props.getProperty(Props.RUN_ID);

    // the output format specifies that the output has to be sorted by
    // query id
    List<KBPEntity> entities = new ArrayList<KBPEntity>();
    entities.addAll(rawRelations.keySet());
    Collections.sort(entities, new KBPEntity.QueryIdSorter());

    // now, for each entity mention, output the block of text as expected
    for (KBPEntity entity : entities) {
      Collection<KBPSlot> mentions = rawRelations.get(entity);

      // first, build a map of the relations we know about...
      Map<String, List<KBPSlot>> mentionMap = new HashMap<String, List<KBPSlot>>();
      for (KBPSlot mention : mentions) {
        String slotName = mention.slotName;
        if (!mentionMap.containsKey(slotName)) mentionMap.put(slotName, new ArrayList<KBPSlot>());
        mentionMap.get(slotName).add(mention);
      }

      for (String slotName : nerTypes.keySet()) {
        // do not output org:* slots for PER and vice versa
        if ((entity.type == EntityType.PERSON && slotName.startsWith("org:"))
            || (entity.type == EntityType.ORGANIZATION && slotName.startsWith("per:"))) {
          // missmatch; move on
          continue;
        }

        List<KBPSlot> slotMentions = mentionMap.get(slotName);
        String outputSlotName = slotName.replaceAll("SLASH", "/");

        String prefix = "";
        if (temporal)
          prefix = entity.queryId.trim() + "\t" + outputSlotName.trim();
        else
          prefix = entity.queryId + " " + outputSlotName + " " + runId;

        String scoreInfo = "";

        if (slotMentions == null || slotMentions.size() == 0) {
          if (!temporal) os.println(prefix + " NIL");
        } else {

          for (KBPSlot mention : slotMentions) {
            if (modelCombinationMode) {
              scoreInfo = " " + mention.getScore();
            }

            if (temporal) {

              if (TSFScore.validSlots.contains(outputSlotName))
                if (mention.t1Slot == null && mention.t2Slot == null && mention.t3Slot == null
                    && mention.t4Slot == null)
                  os.println(prefix + "\tV\t-\t" + runId.trim() + "\t" + mention.docid + "\t" + mention.slotValue);
                else {
                  if (mention.t1Slot != null)
                    os.println(prefix + "\tT1\t" + mention.t1Slot.toISOString().replaceAll("-", "").trim() + "\t"
                        + runId.trim() + "\t" + mention.docid.trim() + "\t" + mention.slotValue);
                  if (mention.t2Slot != null)
                    os.println(prefix + "\tT2\t" + mention.t2Slot.toISOString().replaceAll("-", "").trim() + "\t"
                        + runId.trim() + "\t" + mention.docid.trim() + "\t" + mention.slotValue);
                  if (mention.t3Slot != null)
                    os.println(prefix + "\tT3\t" + mention.t3Slot.toISOString().replaceAll("-", "").trim() + "\t"
                        + runId.trim() + "\t" + mention.docid.trim() + "\t" + mention.slotValue);
                  if (mention.t4Slot != null)
                    os.println(prefix + "\tT4\t" + mention.t4Slot.toISOString().replaceAll("-", "").trim() + "\t"
                        + runId.trim() + "\t" + mention.docid.trim() + "\t" + mention.slotValue);
                }
            } else {
              if(mention.getScore() >= threshold) {
                os.println(prefix + " " + mention.docid + scoreInfo + " " + mention.slotValue);
              }
            }
          }
        }
      }
    }

  }

  private static void prettyPrint(PrintStream os, String header, Map<KBPEntity, Collection<KBPSlot>> relations) {
    os.println(header);
    Set<KBPEntity> entities = relations.keySet();
    for (KBPEntity entity : entities) {
      os.println(entity);
      Collection<KBPSlot> rels = relations.get(entity);
      assert (rels != null);
      for (KBPSlot rel : rels) {
        os.print("\t" + rel.slotName + ":" + rel.slotValue + " (" + rel.getScore() + ")");
        os.println();
      }
    }
  }

  /**
   * Loads in the original query file from the given path. Returns it in the
   * form of a map from entity id to entity.
   */
  public static Map<String, KBPEntity> loadQueryFile(String path) throws IOException, SAXException {
    List<KBPEntity> mentions = TaskXMLParser.parseQueryFile(path);
    Map<String, KBPEntity> originalTask = new HashMap<String, KBPEntity>();
    for (KBPEntity mention : mentions) {
      originalTask.put(mention.id, mention);
    }
    return originalTask;
  }

  public static String normalizeUnicode(String s) {
    // TODO: isn't this a method somewhere in javanlp?
    // TODO: in any case, more things need to be added here
    s = s.replaceAll("&", "&amp;");
    return s;
  }

  public static void makeEntityQueryFile(KBPEntity ent, File f) throws IOException {
    PrintStream os = new PrintStream(new FileOutputStream(f));
    os.println("<?xml version='1.0' encoding='UTF-8'?>");
    os.println("<kbpslotfill>");
    os.println("  <query id=\"" + ent.queryId + "\">");
    os.println("    <name>" + normalizeUnicode(ent.name) + "</name>");
    os.println("    <docid>" + ent.docid + "</docid>");
    os.println("    <enttype>" + ent.type.toString().substring(0, 3) + "</enttype>");
    os.println("    <nodeid>" + ent.id + "</nodeid>");
    if (ent.ignoredSlots != null && ent.ignoredSlots.size() > 0) {
      os.print("    <ignore>");
      boolean first = true;
      for (String ig : ent.ignoredSlots) {
        ig = ig.replace("SLASH", "/");
        if (!first) os.print(" ");
        os.print(ig);
        first = false;
      }
      os.println("</ignore>");
    }
    os.println("  </query>");
    os.println("</kbpslotfill>");
    os.close();
  }

  public Map<KBPEntity, Collection<KBPSlot>> extractEntity(
      KBPEntity testEntity, 
      ListOutput listOutput,
      Map<String, Double> slotNameToThresholds, 
      boolean fillWithFakeDocids, 
      boolean inTuning, 
      int count,
      Properties props,
      Set<String> candidates) throws IOException, SAXException, ParserConfigurationException {
    Log.severe("Answering query: " + testEntity);
    File queryFile = File.createTempFile("onequery", null);
    Log.severe("Using query file: " + queryFile.getAbsolutePath());
    makeEntityQueryFile(testEntity, queryFile);

    Map<KBPEntity, Collection<KBPSlot>> relations = annotateEntity(queryFile.getAbsolutePath(), listOutput,
        slotNameToThresholds, fillWithFakeDocids, inTuning, count, props, candidates);
    queryFile.deleteOnExit();
    return relations;
  }

  /**
   * Removes slot values that are obviously incorrect
   * 
   * @param relations
   */
  private void removeJunk(Map<KBPEntity, Collection<KBPSlot>> relations) {
    Set<KBPEntity> entities = relations.keySet();
    for (KBPEntity ent : entities) {
      Collection<KBPSlot> rels = relations.get(ent);
      Collection<KBPSlot> newRels = new ArrayList<KBPSlot>();

      for (KBPSlot rel : rels) {
        if (validSlot(rel)) {
          newRels.add(rel);
        } else {
          Log.info("Discarding junk slot: " + rel);
        }
      }

      relations.put(ent, newRels);
    }
  }

  private boolean validSlot(KBPSlot rel) {
    //
    // do not keep _NR relations after this step
    //
    if (rel.slotName.equals(RelationMention.UNRELATED)) {
      return false;
    }

    //
    // for now, discard relative dates
    // we detect them heuristically: absolute dates must contain a year
    // specified with digits
    //
    if (KBPSlot.isDateSlot(rel.slotName)) {
      Matcher m = MentionCompatibility.YEAR.matcher(rel.slotValue);
      if (!m.find()) return false;
    }

    // TODO: anything else?

    return true;
  }

  /**
   * Replaces Map.get in the above function, since KBPEntityMention's hashCode
   * actually uses the entity id, not its name.
   * */
  private List<KBPSlot> get(KBPEntity entity, Map<KBPEntity, List<KBPSlot>> entities) {
    for (KBPEntity candidate : entities.keySet()) {
      if (entity.name.equals(candidate.name)) return entities.get(candidate);
    }
    return null;
  }

  /**
   * Checks if two slot values are the similar enough that we should consider
   * the original relation mention to be redundant. To be redundant, the values
   * can either be identical, or the tokens of one can subsume the tokens of the
   * other.
   */
  private boolean matches(String value, String other) {
    if (value.equals(other)) return true;

    List<CoreLabel> tokens = Utils.tokenize(value);
    List<CoreLabel> otherTokens = Utils.tokenize(other);

    if (Utils.contained(tokens, otherTokens, true) || Utils.contained(otherTokens, tokens, true)) return true;

    return false;
  }

  /**
   * Creates a set in which all of the redundant relations from the rawRelations
   * parameter have been removed. Also filters out relations that should be
   * ignored (according to entity.ignoredSlots). See below comments for details
   * on what relations are considered redundant.
   * 
   * @param rawRelations
   * @param kbRelations
   */
  private Map<KBPEntity, Collection<KBPSlot>> removeRedundancies(Map<KBPEntity, Collection<KBPSlot>> rawRelations,
      Map<KBPEntity, List<KBPSlot>> kbRelations) {

    Map<KBPEntity, Collection<KBPSlot>> result = new HashMap<KBPEntity, Collection<KBPSlot>>();

    for (KBPEntity entity : rawRelations.keySet()) {
      Log.severe("IGNORE SLOTS for entity " + entity.id + ": " + entity.ignoredSlots);
      Collection<KBPSlot> relations = rawRelations.get(entity);

      // entity is not even in KB, so keep all associated relation mentions
      // if (get(entity, kbRelations) == null) { result.put(entity, relations); continue; }
      List<KBPSlot> myKbRelations = get(entity, kbRelations);

      Collection<KBPSlot> toKeep = new ArrayList<KBPSlot>();
      for (KBPSlot relation : relations) {
        if (entity.ignoredSlots.contains(relation.slotName)) {
          Log.severe("KB REDUNDANCY: slot will be discarded because it is marked as <ignore>: [" + relation + "].");
          continue;
        } else {
          Log.severe("KB REDUNDANCY: slot is NOT marked as <ignore>: [" + relation + "].");
        }

        boolean redundant = false;
        if (CHECK_AGAINST_KB) {
          SlotToNamedEntities slotInfo = slotsToNamedEntities.getSlotInfo(relation.slotName);
          if (slotInfo == null) throw new RuntimeException("Unknown slot " + relation.slotName);
          if (slotInfo.slotType().equals(SlotType.SINGLE)) {

            // for relations that expect a single value, discard the relation if
            // any relation of the same type exists in the KB, because the KB
            // relation is considered to be a gold-standard
            if (myKbRelations != null) {
              for (KBPSlot kbRelation : myKbRelations) {
                if (kbRelation.slotName.equals(relation.slotName)) {
                  redundant = true;
                  break;
                }
              }
            }
          } else {

            // for list-valued relations, discard the relation if the slot-value
            // is sufficiently similar to an existing relation from the KB
            if (myKbRelations != null) {
              for (KBPSlot kbRelation : myKbRelations) {
                if (matches(relation.slotValue, kbRelation.slotValue)) {
                  redundant = true;
                  break;
                }
              }
            }
          }
        }
        if (!redundant) {
          toKeep.add(relation);
        } else {
          Log.severe("KB REDUNDANCY: slot will be discarded due to KB redundancy: [" + relation + "].");
        }
      }
      if (!toKeep.isEmpty()) result.put(entity, toKeep);
    }
    return result;
  }

  private static boolean CHECK_AGAINST_KB = false;

  /**
   * Keeps LOC_of_death slots only if date_of_death is present
   * 
   * @param relations
   */
  private Map<KBPEntity, Collection<KBPSlot>> handleLocOfDeath(Map<KBPEntity, Collection<KBPSlot>> relations) {
    for (KBPEntity entity : relations.keySet()) {
      // this only applies to PER
      if (entity.type == EntityType.ORGANIZATION) {
        continue;
      }

      Collection<KBPSlot> origSlots = relations.get(entity);
      boolean hasDateOfDeath = false;
      for (KBPSlot r : origSlots) {
        if (r.slotName.equalsIgnoreCase("per:date_of_death")) {
          hasDateOfDeath = true;
          break;
        }
      }

      if (hasDateOfDeath) {
        // if it has date_of_death, we trust the LOC_of_death slots
        continue;
      }

      Collection<KBPSlot> cleanDeathSlots = new ArrayList<KBPSlot>();
      for (KBPSlot r : origSlots) {
        if (!r.slotName.endsWith("_of_death")) {
          cleanDeathSlots.add(r);
        }
      }
      Log.severe("DROPPING ALL LOC_of_death slots for entity " + entity.id + "/" + entity.queryId
          + " because no date_of_death was found.");
      relations.put(entity, cleanDeathSlots);
    }
    return relations;
  }

  @SuppressWarnings("unused")
  private Map<KBPEntity, Collection<KBPSlot>> chooseLongest(Map<KBPEntity, Collection<KBPSlot>> relations) {
    for (KBPEntity entity : relations.keySet()) {
      Collection<KBPSlot> origSlots = relations.get(entity);
      List<KBPSlot> sorted = new ArrayList<KBPSlot>(origSlots);
      Collections.sort(sorted, new Comparator<KBPSlot>() {
        @Override
        public int compare(KBPSlot o1, KBPSlot o2) {
          if (o1.slotValue.length() > o2.slotValue.length()) return -1;
          if (o1.slotValue.length() == o2.slotValue.length()) return 0;
          return 1;
        }
      });
      Collection<KBPSlot> longestOnlySlots = new ArrayList<KBPSlot>();
      for (KBPSlot rel : sorted) {
        KBPSlot longer = null;
        for (KBPSlot other : longestOnlySlots) {
          if (other.slotName.equals(rel.slotName)
              && other.slotValue.toLowerCase().contains(rel.slotValue.toLowerCase())) {
            longer = other;
            break;
          }
        }
        if (longer == null || rel.slotName.equals("per:title")) {
          // no longer slot found; add this one
          // do this for all per:title slots: there are valid titles that are
          // embedded, e.g., "president" and "vice president"
          longestOnlySlots.add(rel);
        } else {
          // found a longer slot; keep that and update its score
          Log.severe("DROPPING SLOT [" + rel + "] because we have a longer one: [" + longer + "]");
          longer.setScore(longer.getScore() + rel.getScore());
        }
      }
      relations.put(entity, longestOnlySlots);
    }
    return relations;
  }

  private Map<KBPEntity, Collection<KBPSlot>> chooseBest(Map<KBPEntity, Collection<KBPSlot>> relations,
      ListOutput listOutput, Map<String, Double> slotNameToThresholds, boolean inTuning, Properties props) {

    // the output will be stored here
    Map<KBPEntity, Collection<KBPSlot>> selectedRelations = new HashMap<KBPEntity, Collection<KBPSlot>>();

    /*
     * we do domain-specific inference under the following conditions: 
     * - if we're not tuning: when doDomainSpecInference is true 
     * - if we're tuning: when doDomainSpecInference is true AND inference.during.tuning is true
     *                    (which is NOT the default)
     */
    boolean inferenceEnabled = !inTuning
        || (inTuning && PropertiesUtils.getBool(props, Props.INFERENCE_DURING_TUNING, false));
    inferenceEnabled = inferenceEnabled && doDomainSpecInference;
    boolean inModelCombination = PropertiesUtils.getBool(props, Props.MODEL_COMBINATION_ENABLED, false);
    Log.severe("inferenceEnabled = " + inferenceEnabled + ", inTuning = " + inTuning + ", inModelCombination = "
        + inModelCombination);

    // performs the actual inference
    KBPInference inference = new KBPInference(getSlotsToNamedEntities(), listOutput, gazetteer, stateAbbreviations,
        stateAbbreviationsToFullName, inferenceEnabled);

    for (KBPEntity entity : relations.keySet()) {
      //
      // first, remove _NR slots and slots under the threshold
      //
      Collection<KBPSlot> filteredSlots = new ArrayList<KBPSlot>();
      for (KBPSlot slot : relations.get(entity)) {
        // keep only slots with score over the threshold
        double threshold = 0;
        String slotName = slot.slotName;
        if (slotNameToThresholds != null) {
          threshold = slotNameToThresholds.get(slotName.replaceAll("SLASH", "/"));
        }
        if (slot.getScore() < threshold) {
          Log.severe("Removed slot [" + slot + "] because its score " + slot.getScore()
              + " is smaller than the acceptance threshold " + threshold);
          continue;
        }
        if (slotName.equals(RelationMention.UNRELATED)) {
          Log.severe("Removed NIL slot: " + slot);
          continue;
        }

        // keep it for the next step
        filteredSlots.add(slot);
      }

      //
      // in model combination mode we keep ALL slots
      // we do not do inference here for model combination
      //
      if (inModelCombination) {
        // TODO: add inference on locations and dates to ModelCombination
        selectedRelations.put(entity, filteredSlots);
        continue;
      }

      //
      // next, run the actual inference
      //
      filteredSlots = inference.inference(filteredSlots); // ed. note: palindrome code!
      selectedRelations.put(entity, filteredSlots);
    }
    return selectedRelations;
  }

  public static void fillDocidsAndFilter(String officialIndex, Map<KBPEntity, Collection<KBPSlot>> relations) {
    for (KBPEntity entity : relations.keySet()) {
      Collection<KBPSlot> mentions = relations.get(entity);
      ArrayList<KBPSlot> newMentions = new ArrayList<KBPSlot>();
      for (KBPSlot mention : mentions) {
        if (mention.docid != null &&
            mention.indexName != null &&
            mention.indexName.equals(officialIndex)) {
          Log.severe("Already found an official doc for slot ["
                  + mention + "] of entity "
                  + entity.toString() + ": "
                  + mention.docid);
          newMentions.add(mention);
          continue;
        }
        Log.severe("DOC FINDER: slot [" + mention + "] will be discarded. No official doc found!");
      }
      mentions.clear();
      mentions.addAll(newMentions);
    }
  }

  public void fillDocidsAndFilter(Map<KBPEntity, Collection<KBPSlot>> relations) throws IOException {
    fillDocidsAndFilter(officialIndex, relations);
  }

  public void fillWithFakeDocid(Map<KBPEntity, Collection<KBPSlot>> relations) {
    for (Collection<KBPSlot> entRels : relations.values()) {
      for (KBPSlot rel : entRels) {
        rel.docid = "DOCID";
        rel.indexName = "INDEX";
      }
    }
  }

  @SuppressWarnings("unused")
  private static String findSlotPOS(NormalizedRelationMention m) {
    CoreMap sent = m.getSentence();
    List<CoreLabel> tokens = sent.get(TokensAnnotation.class);
    assert (tokens != null);
    int start = m.getArg(1).getExtentTokenStart();
    String pos = tokens.get(start).tag();
    return pos;
  }

  @SuppressWarnings("unused")
  private static String chooseMajoritySlotPOS(Collection<NormalizedRelationMention> mentions) {
    Counter<String> posCounts = new ClassicCounter<String>();
    String firstPOS = null;

    for (RelationMention m : mentions) {
      CoreMap sent = m.getSentence();
      List<CoreLabel> tokens = sent.get(TokensAnnotation.class);
      assert (tokens != null);
      int start = m.getArg(1).getExtentTokenStart();
      String pos = tokens.get(start).tag();
      posCounts.incrementCount(pos);
      if (firstPOS == null) {
        firstPOS = pos;
      }
    }

    List<Pair<String, Double>> sortedCounts = Counters.toDescendingMagnitudeSortedListWithCounts(posCounts);
    double firstCount = posCounts.getCount(firstPOS);
    if (firstCount == sortedCounts.get(0).second())
      return firstPOS;
    else
      return sortedCounts.get(0).first();
  }

  private boolean overlappingPossible(String newSlot, Set<String> usedSlots) {
    for (String oldSlot : usedSlots) {
      if (!overlappingSlot(newSlot, oldSlot)) {
        return false;
      }
    }
    return true;
  }
  
  /**
   * Clusters together relation mentions that contain the same <entity, slot> pair
   * This is needed for the joint inference
   * @param dms
   * @return
   */
  private List<List<DatumAndMention>> clusterMentions(List<DatumAndMention> dms) {
    List<List<DatumAndMention>> clusters = new ArrayList<List<DatumAndMention>>();
    
    for (int crt = 0; crt < dms.size(); crt++) {
      DatumAndMention dm = dms.get(crt);
      boolean found = false;
      for (List<DatumAndMention> cluster : clusters) {
        for (DatumAndMention otherDm : cluster) {
          if (mentionCompatibility.mergeable(dm.mention(), otherDm.mention())){
            found = true;
            break;
          }
        }
        if (found) {
          cluster.add(dm);
          break;
        }
      }
      if (!found) {
        List<DatumAndMention> cluster = new ArrayList<DatumAndMention>();
        cluster.add(dm);
        clusters.add(cluster);
      }
    }
    
    return clusters;
  }
  
  /**
   * Reads all the slot candidates for one entity
   * Each <entity, slot value> yields a different KBPTuple, which contains all the mentions of the slot in text
   */
  private List<KBPTuple> readTuples(
      String queryFile,
      Map<String, KBPEntity> originalTask,
      Set<String> candidates) throws IOException, SAXException {

    // load from cache if available
    // this is useful because we can run the evaluator wo/ access to the index
    Map<KBPEntity, List<KBPSlot>> entities = KBPReader.parseQueryFile(queryFile);
    if(entities.size() != 1)
      throw new RuntimeException("ERROR: readTuples works for a single entity but I found " + entities.size() + "!");
    String queryId = entities.keySet().iterator().next().queryId;
    List<KBPTuple> tuples = QueryCacher.load(workDir + File.separator + "test", queryId);
    if(tuples != null){
      Log.severe("Loaded " + tuples.size() + " tuples from cache for query " + queryId);
      // save candidates for our restricted scoring
      for(KBPTuple tuple: tuples) {
        for(String ns: tuple.normalizedSlotValues()) {
          String candKey = queryId + ":" + ns;
          candidates.add(candKey);
          Log.severe("FOUND SLOT CANDIDATE: " + candKey);
        }
      }
      return tuples;
    }

    //
	  // Extract all sentences that contain mentions of the entities in the test queries
	  //
	  Annotation testSentences = reader.parse(queryFile);
	  Log.severe("TIMER: KBPReader.parse complete.");

	  //
    // Convert RelationMentions to datums
    // This implements both the mention and the relation model (where compatible
    // mentions are merged into a single datum)
    //
    Counter<String> localDomainStats = new ClassicCounter<String>();
    List<DatumAndMention> dms = reader.generateDatums(testSentences, rff, null, localDomainStats);
    Log.severe("Constructed " + dms.size() + " datums for the given queries.");
    Log.severe("DOMAIN STATS for query file " + queryFile + ": " + localDomainStats);
    domainStats.addAll(localDomainStats);
    
    //
    // Save candidates
    //
    if(candidates != null){
      for(DatumAndMention dm: dms) {
        EntityMention em = (EntityMention) dm.mention().getArg(0);
        // String queryId = getQueryId(em, originalTask);
        String slot = dm.mention().getNormalizedSlotValue();
        String candKey = queryId + ":" + slot;
        candidates.add(candKey);
        Log.severe("FOUND SLOT CANDIDATE: " + candKey);
      }
    }
    
    //
    // Cluster all mentions that contain the same <entity, slot> pairs
    //
    List<List<DatumAndMention>> relations = clusterMentions(dms);
    
    //
    // Build the tuple objects
    //
    tuples = new ArrayList<KBPTuple>();
    for(List<DatumAndMention> relation: relations) {
      tuples.add(new KBPTuple(relation, originalTask, docidFinder));
    }

    // store in cache
    QueryCacher.save(workDir + File.separator + "test", queryId, tuples);
    return tuples;
  }

  /**
   * Finds raw instances of slots for the entities of interest in all our indices
   * 
   * @param queryFile File containing info on the input entity
   * @param candidates Stores all candidate slots found for this entity (needed for scoring facets)
   * @return All relations found in the data (need cleanup to match KBP specs!)
   * @throws IOException
   */
  private Map<KBPEntity, Collection<KBPSlot>> annotateRaw(String queryFile, Set<String> candidates) throws IOException, SAXException {
    //
    // Read slot candidates from the index
    //
    Map<String, KBPEntity> originalTask = loadQueryFile(queryFile);
    List<KBPTuple> tuples = readTuples(queryFile, originalTask, candidates); 

    //
    // Classify all relations
    //
    List<Counter<String>> predictedLabels = new ArrayList<Counter<String>>();

    //
    // In diagnostic mode keep the true labels from the RelationMention,
    // do not call the classifier
    //
    if (diagnosticMode) {
      for (KBPTuple tuple: tuples) {
        Counter<String> labels = new ClassicCounter<String>();
        for(int i = 0; i < tuple.size(); i ++){
          if (tuple.goldMentionLabel(i).equals(RelationMention.UNRELATED)) {
            throw new RuntimeException("ERROR: cannot have a NIL relation in diagnostic mode!");
          }
          labels.incrementCount(tuple.goldMentionLabel(i), 1.0);
        }
        predictedLabels.add(labels);
      }
    } else {
      // not diagnostic mode, doing actual classification
      // call the distant-supervision classifier
      for (KBPTuple tuple: tuples) {
        // raw labels assigned by the classifier
        // note: the NIL MUST not be included in this set
        // note: the labels MUST be sorted in descending order of probability
        Counter<String> labels = relationExtractor.classifyRelation(tuple);
        List<Pair<String, Double>> sortedLabels = Counters.toDescendingMagnitudeSortedListWithCounts(labels);

        // keep only the labels that are valid according to the domain definition
        Counter<String> filteredLabels = new ClassicCounter<String>();

        Set<String> usedLabels = new HashSet<String>();
        for (Pair<String, Double> labelProb : sortedLabels) {
          if (overlappingPossible(labelProb.first(), usedLabels) &&
              SlotValidity.validCandidateForLabel(slotsToNamedEntities, 
                  labelProb.first(), tuple.entityType(), tuple.slotType())) {
            filteredLabels.setCount(labelProb.first(), labelProb.second());
            usedLabels.add(labelProb.first());
          }
        }
        predictedLabels.add(filteredLabels);
      }
    }

    //
    // Convert to KBP data structures
    //
    return annotationsToKBPStructures(originalTask, tuples, predictedLabels);
  }
  
  /**
   * Converts the RelationMention objects created by our model to
   * KBPRelationMentions (one set for each known entity)
   * 
   */
  private Map<KBPEntity, Collection<KBPSlot>> annotationsToKBPStructures(
      Map<String, KBPEntity> originalTask,
      List<KBPTuple> tuples,
      List<Counter<String>> labels) {

    assert(tuples.size() == labels.size());
    Map<KBPEntity, Collection<KBPSlot>> map = new HashMap<KBPEntity, Collection<KBPSlot>>();

    for(int which = 0; which < tuples.size(); which ++){
      KBPTuple tuple = tuples.get(which);
      Counter<String> relLabels = labels.get(which);
      KBPEntity kbpEnt = makeKbpEntity(tuple, originalTask);
      
      Collection<KBPSlot> kbpRelations = map.get(kbpEnt);
      if (kbpRelations == null) {
        kbpRelations = new HashSet<KBPSlot>();
        map.put(kbpEnt, kbpRelations);
      }
      
      // convert this relation to one or more KBPRelationMention
      List<KBPSlot> kbpRels = makeKbpRelations(tuple, relLabels, kbpEnt);

      kbpRelations.addAll(kbpRels);
    }

    return map;
  }
  
  private String getQueryId(EntityMention em, Map<String, KBPEntity> originalTask) {
    assert (em != null);
    KBPEntity originalQuery = originalTask.get(KBPTuple.getKbpId(em));
    return originalQuery.queryId;
  }

  private KBPEntity makeKbpEntity(KBPTuple tuple, Map<String, KBPEntity> originalTask) {
    EntityType type = tuple.entityKBPType();
    String id = tuple.entityId();
    KBPEntity kbpEnt = new KBPEntity();
    kbpEnt.id = id;
    kbpEnt.name = tuple.entityName();
    kbpEnt.type = type;
    KBPEntity originalQuery = originalTask.get(kbpEnt.id);
    if (originalQuery != null) {
      kbpEnt.ignoredSlots = originalQuery.ignoredSlots;
      kbpEnt.queryId = originalQuery.queryId;
    }

    return kbpEnt;
  }
  
  private KBPSlot makeKBPRelationMention(String slotType, 
      String slotValue, 
      double score, 
      KBPEntity entity) {
    KBPSlot kbpRel = new KBPSlot(entity.name, entity.id, slotValue, slotType);
    kbpRel.setScore(score);
    return kbpRel;
  }
  
  private String normalizeSlotValue(String slotValue, String candidateNE, String slotName) {
    // if the type chosen prefers COUNTRY NEs and the NE label of the candidate
    // is NATIONALITY => change to COUNTRY
    if (Constants.COUNTRY_EQ_NATIONALITY) {
      if (KBPSlot.isPureCountryNameSlot(slotName) && candidateNE.equals("NATIONALITY")) {
        String countryName = nationalities.nationalityToCountry(slotValue);
        if (countryName != null) {
          Log.severe("CHANGED NATIONALITY " + slotValue + " to country name " + countryName);
          return countryName;
        }
      }
    }

    return slotValue;
  }

  private List<KBPSlot> makeKbpRelations(
      KBPTuple tuple,
      Counter<String> labels,
      KBPEntity entity) {
    //
    // make one KBPSlot for each label proposed
    //

    // sanity check: make sure we do not store the NIL label here
    assert(! labels.keySet().contains(RelationMention.UNRELATED));
    
    // TODO: generate the temporal attributes for this relation
    // this must replicate the functionality from ConflictResolutionForMentionModel.chooseSlots()
    // the last release that contained the ConflictResolutionForMentionModel class was 38387. Look there for code.
    
    // create the actual KBPSlots; one per each key in labels 
    List<KBPSlot> rels = new ArrayList<KBPSlot>();
    if(labels.size() == 0){
      // NIL relation
      String slotType = RelationMention.UNRELATED;
      double score = 1.0;
      rels.add(makeKBPRelationMention(slotType, tuple.slotValue(), score, entity));
    } else {
      // one or more non-NIL relations
      List<Pair<String, Double>> sortedScores = 
        Counters.toDescendingMagnitudeSortedListWithCounts(labels);
      for (Pair<String, Double> typeAndScore : sortedScores) {
        String realSlotValue = 
          normalizeSlotValue(tuple.slotValue(), tuple.slotType(), typeAndScore.first());
        String slotName = typeAndScore.first();
        rels.add(makeKBPRelationMention(slotName, realSlotValue, 
            typeAndScore.second(), entity)); 
      }
    }

    for (KBPSlot kbpRel : rels) {
      kbpRel.docid = tuple.docid();
      kbpRel.indexName = tuple.indexName();
    }

    return rels;
  }

  public Map<KBPEntity, Collection<KBPSlot>> annotateEntity(
      String queryFile, 
      ListOutput listOutput,
      Map<String, Double> slotNameToThresholds, 
      boolean fillWithFakeDocids, 
      boolean inTuning, 
      int count,
      Properties props,
      Set<String> candidates) throws IOException, SAXException, ParserConfigurationException {
    
    // These are the raw relations, obtained simply by running the classifier over the whole data
    Map<KBPEntity, Collection<KBPSlot>> relations = annotateRaw(queryFile, candidates);
    Log.severe("TIMER: annotateRaw complete.");
    prettyPrint(System.err, "Raw relations:", relations);

    // remove slots that are obviously incorrect, e.g., dates wo any digits and _NR
    removeJunk(relations);
    Log.severe("TIMER: removeJunk complete.");

    if (fillWithFakeDocids) {
      Log.severe("TIMER: not calling removeRedundancies.");
    } else {
      // Remove slots that are marked as ignored in the query file
      relations = removeRedundancies(relations, kbEntities);
      prettyPrint(System.err, "Relations after removing slots redundant with the KB", relations);
      Log.severe("TIMER: removeRedundancies complete.");
    }

    // Pick the best candidates for each slot
    relations = chooseBest(relations, listOutput, slotNameToThresholds, inTuning, props);
    prettyPrint(System.err, "Relations after picking best per slot", relations);
    Log.severe("TIMER: chooseBest complete.");

    // For list slots, keep only the longest value found for embedded slots
    // relations = chooseLongest(relations);
    // prettyPrint(System.err, "Relations after picking longest value per overlapping slots", relations);
    // Log.severe("TIMER: chooseLongest complete.");

    // Keep LOC_of_death only of date_of_death exists
    // This is needed because the noise level on LOC_of_death is very high, and
    // the one on date_of_death is low
    relations = handleLocOfDeath(relations);
    prettyPrint(System.err, "Relations after cleaning per:*_of_death slots", relations);
    Log.severe("TIMER: handleLocOfDeath complete.");

    if (fillWithFakeDocids) {
      Log.severe("Filling with fake docids");
      fillWithFakeDocid(relations);
    } else {
      Log.severe("Filling with real docids and filtering");
      fillDocidsAndFilter(relations);
    }

    prettyPrint(System.err, "Relations after doc finding:", relations);
    Log.severe("TIMER: fillDocidsAndFilter complete.");

    return relations;
  }

  public Map<KBPEntity, Collection<KBPSlot>> annotate(
      String queryFile, 
      ListOutput listOutput,
      Map<String, Double> slotNameToThresholds, 
      boolean fillWithFakeDocids, 
      boolean inTuning, 
      Properties props,
      Set<String> candidates)
      throws IOException, SAXException, ParserConfigurationException {
    Map<KBPEntity, Collection<KBPSlot>> allSlots = new HashMap<KBPEntity, Collection<KBPSlot>>();

    //
    // the entities in the query file
    //
    Map<String, KBPEntity> originalTask = loadQueryFile(queryFile);

    // extract one entity at a time
    int count = 0;
    for (String entityId : originalTask.keySet()) {
      count++;
      System.out.println("entity id in annotate function is " + entityId);
      KBPEntity testEntity = originalTask.get(entityId);
      System.out.println("entity mention in annotate function is  " + testEntity);
      Map<KBPEntity, Collection<KBPSlot>> entSlots = extractEntity(
          testEntity, 
          listOutput, 
          slotNameToThresholds,
          fillWithFakeDocids, 
          inTuning, 
          count, 
          props,
          candidates);
      for (KBPEntity ent : entSlots.keySet()) {
        allSlots.put(ent, entSlots.get(ent));
      }
    }
    
    // make sure ALL entities are stored in this map, even if no slots are found
    // this is crucial for correct scoring (otherwise, entities with no slots are skipped)
    for(KBPEntity ent: originalTask.values()) {
      if(! allSlots.containsKey(ent)){
        allSlots.put(ent, new ArrayList<KBPSlot>());
      }
    }

    Log.severe("OVERALL DOMAIN STATS: " + domainStats);
    return allSlots;
  }

  /**
   * Keeps only the relations predicted with a score larger than threshold The
   * original relations are not modified in any way.
   */
  private static Map<KBPEntity, Collection<KBPSlot>> filterByThreshold(Map<KBPEntity, Collection<KBPSlot>> relations,
      double threshold) {
    Set<KBPEntity> entities = relations.keySet();
    Map<KBPEntity, Collection<KBPSlot>> filteredRelations = new HashMap<KBPEntity, Collection<KBPSlot>>();
    for (KBPEntity entity : entities) {
      Collection<KBPSlot> myRelations = relations.get(entity);
      Collection<KBPSlot> myFilteredRels = new ArrayList<KBPSlot>();
      for (KBPSlot rel : myRelations) {
        if (rel.getScore() >= threshold) {
          myFilteredRels.add(rel);
        }
      }
      filteredRelations.put(entity, myFilteredRels);
    }
    return filteredRelations;
  }

  /**
   * Keeps only the relations of a specific slot name. The original relations
   * are not modified in any way.
   */
  private static Map<KBPEntity, Collection<KBPSlot>> filterBySlotName(Map<KBPEntity, Collection<KBPSlot>> relations,
      String slotName) {
    Set<KBPEntity> entities = relations.keySet();
    Map<KBPEntity, Collection<KBPSlot>> filteredRelations = new HashMap<KBPEntity, Collection<KBPSlot>>();
    for (KBPEntity entity : entities) {
      Collection<KBPSlot> myRelations = relations.get(entity);
      Collection<KBPSlot> myFilteredRels = new ArrayList<KBPSlot>();
      for (KBPSlot rel : myRelations) {
        if (rel.slotName.equals(slotName)) {
          myFilteredRels.add(rel);
        }
      }
      filteredRelations.put(entity, myFilteredRels);
    }
    return filteredRelations;
  }

  /**
   * Determine the best empirical tuning threshold for slots. If slotName is
   * null, we will determine the best threshold for all slots. Otherwise, we
   * will restrict ourselves only to those relations of type slotName.
   */
  private static double tuneThreshold(Properties props, ListOutput listOutput, String slotName) throws Exception {
    double bestThreshold = -1;
    double bestF1 = Double.MIN_VALUE;
    double bestP = Double.MIN_VALUE;
    double bestR = Double.MIN_VALUE;
    String prefix = "TUNING: " + (slotName == null ? "" : "(" + slotName + ") ");
    Log.severe(prefix + "started...");

    // run the system with a very inclusive threshold
    KBPEvaluator tester = new KBPEvaluator(props);
    String queryFile = props.getProperty(Props.DEV_QUERIES);
    boolean fillWithFakeDocids = !PropertiesUtils.getBool(props, Props.DOC_FINDING_DURING_TUNING, true);
    boolean anydoc = PropertiesUtils.getBool(props, Props.ANYDOC, Constants.DEFAULT_ANYDOC);
    Set<String> allCandidates = new HashSet<String>();
    Map<KBPEntity, Collection<KBPSlot>> relations = 
      tester.annotate(queryFile, listOutput, null, fillWithFakeDocids, true, props, allCandidates);
    Set<String> queryIds = extractQueryIds(relations.keySet());

    // Set<String> slots = null;
    if (slotName != null) {
      relations = filterBySlotName(relations, slotName);
      // relation specific scoring is busted for now
      // slots = new HashSet<String>();
      // slots.add(slotName.replaceAll("SLASH", "/"));

    }

    for (double singleThreshold = 0.00; singleThreshold <= 10.00; singleThreshold += 0.10) {
      Map<KBPEntity, Collection<KBPSlot>> filteredRelations = filterByThreshold(relations, singleThreshold);

      // generate scorable output
      String workDir = props.getProperty(Props.WORK_DIR);
      String runid = props.getProperty(Props.RUN_ID);
      assert (runid != null);
      String outputFileName = workDir + File.separator + runid + ".dev.output";
      File outputFile = new File(outputFileName);
      outputFile.deleteOnExit();
      PrintStream os = new PrintStream(outputFile);
      outputRelations(os, props, tester.getSlotsToNamedEntities(), filteredRelations, false);
      os.close();

      // score using the official scorer
      String keyFile = props.getProperty(Props.GOLD_RESPONSES);
      assert (keyFile != null);
      String scoreFileName = workDir + File.separator + runid + "." + props.getProperty(Props.QUERY_SCORE_FILE)
          + ".dev.txt";
      File scoreFile = new File(scoreFileName);
      scoreFile.deleteOnExit();
      PrintStream sos = new PrintStream(new FileOutputStream(scoreFile));
      Pair<Double, Double> score = SFScore.score(sos, outputFileName, keyFile, null, anydoc, allCandidates, queryIds);
      double f1 = SFScore.pairToFscore(score);
      sos.close();
      Log.severe(prefix + "F1 score for threshold " + singleThreshold + " is " + f1 + "(P " + score.first + ", R "
          + score.second + ")");

      // best so far?
      if (f1 > bestF1) {
        bestThreshold = singleThreshold;
        bestF1 = f1;
        bestP = score.first();
        bestR = score.second();
        Log.severe(prefix + "found current best F1: " + bestF1);
      }
    }

    String suffix = "";
    if (bestThreshold == -1) {
      suffix = " (didn't find any useful threshold settings, using default)";
      bestThreshold = Constants.DEFAULT_SLOT_THRESHOLD;
    }

    Log.severe(prefix + "selected final threshold " + bestThreshold + " with P " + bestP + " R " + bestR + " F1 "
        + bestF1 + suffix);
    return bestThreshold;
  }

  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    Log.setLevel(Log.stringToLevel(props.getProperty(Props.LOG_LEVEL, "INFO")));
    Log.severe("Using properties: " + props);

    // enable coref during testing!
    props.setProperty(Props.INDEX_PIPELINE_METHOD, "FULL");

    // convert path names to use the local machine name
    // props.setProperty(Props.INDEX_PROP,
    // Utils.convertToHostName(props.getProperty(Props.INDEX_PROP)));
    // Log.severe("Actual " + Props.INDEX_PROP + " property used: " +
    // props.getProperty(Props.INDEX_PROP));
    // props.setProperty(Props.WEB_CACHE_PROP,
    // Utils.convertToHostName(props.getProperty(Props.WEB_CACHE_PROP)));
    // Log.severe("Actual " + Props.WEB_CACHE_PROP + " property used: " +
    // props.getProperty(Props.WEB_CACHE_PROP));
    // props.setProperty(Props.OFFICIAL_INDEX_PROP,
    // Utils.convertToHostName(props.getProperty(Props.OFFICIAL_INDEX_PROP)));
    // Log.severe("Actual " + Props.OFFICIAL_INDEX_PROP + " property used: " +
    // props.getProperty(Props.OFFICIAL_INDEX_PROP));

    boolean modelCombinationEnabled = PropertiesUtils.getBool(props, Props.MODEL_COMBINATION_ENABLED, false);

    // turn it off for now so it doesn't disrupt outputs in chooseBest()
    props.setProperty(Props.MODEL_COMBINATION_ENABLED, "false");
    List<Double> thresholds = extractAndScore(props, false);

    if (modelCombinationEnabled) {
      Log.info("Re-extracting and scoring for model combination");
      props.setProperty(Props.SLOT_THRESHOLD_PER_RELATION, "true");
      props.setProperty(Props.SLOT_THRESHOLD, StringUtils.join(thresholds, ","));
      props.setProperty(Props.MODEL_COMBINATION_ENABLED, "true");
      extractAndScore(props, true);
    }
  }

  public static List<Double> extractAndScore(Properties props, boolean modelCombinationMode) throws Exception,
      IOException, SAXException, ParserConfigurationException, FileNotFoundException {
    String listOutputProp = props.getProperty(Props.LIST_OUTPUT, "all");
    ListOutput listOutput = null;
    if (listOutputProp.equalsIgnoreCase("all"))
      listOutput = ListOutput.ALL;
    else if (listOutputProp.equalsIgnoreCase("best"))
      listOutput = ListOutput.BEST;
    else
      throw new RuntimeException("Unknown value for the kbp.list.output property: " + listOutputProp);
    Log.info("Strategy for list slots is: " + listOutput);

    List<String> allSlotNames = new ArrayList<String>(SFScore.allSlots);
    Collections.sort(allSlotNames);

    Map<String, Double> slotNameToThresholds = new HashMap<String, Double>();
    double singleThreshold = -1;
    boolean slotThresholdPerRelation = PropertiesUtils.getBool(props, Props.SLOT_THRESHOLD_PER_RELATION);
    boolean anydoc = PropertiesUtils.getBool(props, Props.ANYDOC, Constants.DEFAULT_ANYDOC);
    Log.info("When scoring, accept any doc: " + anydoc);

    if (slotThresholdPerRelation) {
      if (props.containsKey(Props.SLOT_THRESHOLD)) {
        // load the thresholds from props:
        // the thresholds must be stored as a list of comma-separated values,
        // one threshold for each slot name.
        // the order of slot names is alphanumeric.
        double[] thresholds = PropertiesUtils.getDoubleArray(props, Props.SLOT_THRESHOLD);
        assert thresholds.length == allSlotNames.size();
        int thresholdIndex = 0;
        for (String slotName : allSlotNames) {
          slotNameToThresholds.put(slotName, thresholds[thresholdIndex++]);
        }
      } else {
        // estimate a threshold for each slot name
        for (String slotName : allSlotNames) {
          double perSlotThreshold = tuneThreshold(props, listOutput, slotName);

          slotNameToThresholds.put(slotName, perSlotThreshold);
        }
      }
    } else {
      if (props.containsKey(Props.SLOT_THRESHOLD)) {
        singleThreshold = PropertiesUtils.getDouble(props, Props.SLOT_THRESHOLD);
      } else {
        singleThreshold = tuneThreshold(props, listOutput, null);
      }
      Log.info("Threshold for ALL slots is: " + singleThreshold);

      // store as the threshold for all slots
      for (String slotName : allSlotNames) {
        slotNameToThresholds.put(slotName, singleThreshold);
      }
    }

    KBPEvaluator tester = new KBPEvaluator(props);
    String queryFile = props.getProperty(Props.TEST_QUERIES);
    Set<String> allCandidates = new HashSet<String>();
    Map<KBPEntity, Collection<KBPSlot>> relations = 
      tester.annotate(queryFile, listOutput, slotNameToThresholds, false, false, props, allCandidates);

    prettyPrint(System.err, "Final relations (modelCombinationMode=" + modelCombinationMode + ")", relations);
    outputRelations(System.err, props, tester.getSlotsToNamedEntities(), relations, modelCombinationMode);

    String workDir = props.getProperty(Props.WORK_DIR);
    String runid = props.getProperty(Props.RUN_ID);
    assert (runid != null);
    
    if(props.containsKey(Props.PERCEPTRON_THRESHOLD)) {
      int thr = (int) (PropertiesUtils.getDouble(props, Props.PERCEPTRON_THRESHOLD) * 100.0);
      runid += "_t" + Integer.toString(thr);
    }
    
    String outputFileName = workDir + File.separator + runid + ".output";
    if (modelCombinationMode) {
      outputFileName += ".combo";
    }
    PrintStream os = new PrintStream(outputFileName);
    outputRelations(os, props, tester.getSlotsToNamedEntities(), relations, modelCombinationMode);
    os.close();
    System.err.println("The output was also saved in file: " + outputFileName);

    String keyFile = props.getProperty(Props.GOLD_RESPONSES);

    if (temporal) {
      cleanOutputFile(keyFile, outputFileName, runid);
      System.err.println("The cleaned output was saved in file: " + outputFileName + "_cleaned");
    }

    boolean scoreTestQueries = PropertiesUtils.getBool(props, Props.SCORE_TEST, true);

    // no need to rescore for model combination mode
    if (!modelCombinationMode && scoreTestQueries == true) {
      //
      // score using the official scorer
      // keyFile must be in 2010 format. If it's in 2009 format, use UpdateSFKey to convert to 2010
      //
      if (keyFile != null) {
        System.out.println("Official KBP score:");
        Set<String> queryIds = extractQueryIds(relations.keySet());
        if (!temporal) SFScore.score(System.out, outputFileName, keyFile, anydoc, allCandidates, queryIds);

        int threshold = (int) (100.0 * singleThreshold);
        String scoreFileName = workDir + File.separator + runid + "." + props.getProperty(Props.QUERY_SCORE_FILE)
            + "_t" + threshold + ".txt";
        PrintStream sos = new PrintStream(new FileOutputStream(scoreFileName));
        if (temporal)
          TSFScore.temporalScore(outputFileName, keyFile, Boolean.parseBoolean(props.getProperty(
              Props.TEMPORAL_RESULTTRACE, "false")), true);
        else
          SFScore.score(sos, outputFileName, keyFile, anydoc, allCandidates, queryIds);
        sos.close();

      }
    }
    
    // this generates the scores for the P/R curve
    if(PropertiesUtils.getBool(props, Props.SHOW_CURVE, true)) {
      // generatePRCurve(props, tester, relations, allCandidates);
      generatePRCurveNonProbScores(props, tester, relations, allCandidates);
    }

    List<Double> thresholds = new ArrayList<Double>();
    for (String slot : allSlotNames) {
      thresholds.add(slotNameToThresholds.get(slot));
    }
    return thresholds;
  }
  
  private static Set<String> extractQueryIds(Set<KBPEntity> ents) {
    Set<String> qids = new HashSet<String>();
    for(KBPEntity e: ents) qids.add(e.queryId);
    return qids;
  }

  private static void generatePRCurve(Properties props,
      KBPEvaluator tester,
      Map<KBPEntity, Collection<KBPSlot>> relations,
      Set<String> allCandidates) throws IOException {
    String workDir = props.getProperty(Props.WORK_DIR);
    String runid = props.getProperty(Props.RUN_ID);
    assert (runid != null);
    
    File dir = new File(workDir + File.separator + runid + ".prcurve");
    dir.mkdir();
    
    String keyFile = props.getProperty(Props.GOLD_RESPONSES);
    assert(keyFile != null);
    Set<String> queryIds = extractQueryIds(relations.keySet());
    boolean anydoc = PropertiesUtils.getBool(props, Props.ANYDOC, Constants.DEFAULT_ANYDOC);
    
    PrintStream mos = new PrintStream(dir + File.separator + runid + ".curve");
    for(double t = 1.0; t >= 0; ) {
      String outputFileName = dir + File.separator + runid + ".t" + t + ".output";
      PrintStream os = new PrintStream(outputFileName);
      outputRelations(os, props, tester.getSlotsToNamedEntities(), relations, false, t);
      os.close();
      
      String scoreFileName = dir + File.separator + runid + ".t" + t + ".score";
      PrintStream sos = new PrintStream(new FileOutputStream(scoreFileName));
      Pair<Double, Double> pr = SFScore.score(sos, outputFileName, keyFile, anydoc, allCandidates, queryIds);
      double f1 = (pr.first() != 0 && pr.second() != 0 ? 2*pr.first()*pr.second()/(pr.first()+pr.second()) : 0.0);
      sos.close();
      
      mos.println(t + " P " + pr.first() + " R " + pr.second() + " F1 " + f1);
      
      // these increments are identical to those in MultiR.generatePRCurve()
      if(t > 1.0) t -= 1.0;
      else if(t > 0.99) t -= 0.0001;
      else if(t > 0.95) t -= 0.001;
      else t -= 0.01;
    }
    mos.close();
  }

  private static void generatePRCurveNonProbScores(Properties props,
                                      KBPEvaluator tester,
                                      Map<KBPEntity, Collection<KBPSlot>> relations,
                                      Set<String> allCandidates) throws IOException {
    String workDir = props.getProperty(Props.WORK_DIR);
    String runid = props.getProperty(Props.RUN_ID);
    assert (runid != null);

    File dir = new File(workDir + File.separator + runid + ".prcurve.tmp");
    dir.mkdir();

    String keyFile = props.getProperty(Props.GOLD_RESPONSES);
    assert(keyFile != null);
    Set<String> queryIds = extractQueryIds(relations.keySet());
    boolean anydoc = PropertiesUtils.getBool(props, Props.ANYDOC, Constants.DEFAULT_ANYDOC);

    String prFileName = workDir + File.separator + runid + ".curve";
    PrintStream mos = new PrintStream(prFileName);
    List<Pair<KBPEntity, KBPSlot>> sorted = convertToSorted(relations);
    int START_OFFSET = 10;

    for(int i = START_OFFSET; i < sorted.size(); i ++) {
      Map<KBPEntity, Collection<KBPSlot>> filteredRels = keepTop(sorted, i);
      String outputFileName = dir + File.separator + runid + ".i" + i + ".output";
      PrintStream os = new PrintStream(outputFileName);
      outputRelations(os, props, tester.getSlotsToNamedEntities(), filteredRels, false, 0);
      os.close();

      String scoreFileName = dir + File.separator + runid + ".i" + i + ".score";
      PrintStream sos = new PrintStream(new FileOutputStream(scoreFileName));
      Pair<Double, Double> pr = SFScore.score(sos, outputFileName, keyFile, anydoc, allCandidates, queryIds);
      double f1 = (pr.first() != 0 && pr.second() != 0 ? 2*pr.first()*pr.second()/(pr.first()+pr.second()) : 0.0);
      sos.close();

      double ratio = (double) i / (double) sorted.size();
      mos.println(ratio + " P " + pr.first() + " R " + pr.second() + " F1 " + f1);
    }
    mos.close();
    Log.severe("P/R curve data generated in file: " + prFileName);

    // let's remove the tmp dir with partial scores. we are unlikely to need all this data.
    File [] tmpFiles = dir.listFiles();
    boolean deleteSuccess = true;
    for(File f: tmpFiles) {
      deleteSuccess = deleteSuccess && f.delete();
    }
    deleteSuccess = deleteSuccess && dir.delete();
    if(! deleteSuccess) {
      Log.severe("Tried to delete P/R tmp directory but failed: " + dir.getAbsolutePath());
    }
  }

  private static List<Pair<KBPEntity, KBPSlot>> convertToSorted(Map<KBPEntity, Collection<KBPSlot>> relations) {
    List<Pair<KBPEntity, KBPSlot>> sorted = new ArrayList<Pair<KBPEntity, KBPSlot>>();
    for(KBPEntity e: relations.keySet()) {
      for(KBPSlot s: relations.get(e)) {
        sorted.add(new Pair<KBPEntity, KBPSlot>(e, s));
      }
    }
    Collections.sort(sorted, new Comparator<Pair<KBPEntity, KBPSlot>>() {
      @Override
      public int compare(Pair<KBPEntity, KBPSlot> s1, Pair<KBPEntity, KBPSlot> s2) {
        if(s1.second().getScore() > s2.second().getScore()) return -1;
        if(s1.second().getScore() < s2.second().getScore()) return 1;
        return 0;
      }
    });
    return sorted;
  }
  private static Map<KBPEntity, Collection<KBPSlot>> keepTop(List<Pair<KBPEntity, KBPSlot>> sorted, int end) {
    Map<KBPEntity, Collection<KBPSlot>> filtered = new HashMap<KBPEntity, Collection<KBPSlot>>();
    for(int i = 0; i < end && i < sorted.size(); i ++) {
      KBPEntity e = sorted.get(i).first();
      KBPSlot s = sorted.get(i).second();
      Collection<KBPSlot> slots = filtered.get(e);
      if(slots == null) {
        slots = new ArrayList<KBPSlot>();
        filtered.put(e, slots);
      }
      slots.add(s);
    }
    return filtered;
  }
}