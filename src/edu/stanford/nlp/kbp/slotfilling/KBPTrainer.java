package edu.stanford.nlp.kbp.slotfilling;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileOutputStream;
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
import java.util.Random;
import java.util.Set;

import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.classify.WeightedDataset;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.io.FileSystem;
import edu.stanford.nlp.kbp.slotfilling.classify.JointlyTrainedRelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.classify.ModelType;
import edu.stanford.nlp.kbp.slotfilling.classify.MultiLabelDataset;
import edu.stanford.nlp.kbp.slotfilling.classify.OneVsAllRelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationExtractorFactory;
import edu.stanford.nlp.kbp.slotfilling.common.Constants;
import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.KBPSlot;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.MinimalDatum;
import edu.stanford.nlp.kbp.slotfilling.common.ProcessWrapper;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.kbp.slotfilling.common.RelationDatum;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.KBPDomReader;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.KBPReader;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.StringUtils;

/**
 * Trains the relex model for KBP
 * This has the same functionality as MachineReading, but uses a LOT less memory
 */
public class KBPTrainer {
  private static final boolean TEST_ON_DEVEL = false;
  private static final boolean EVALUATE_AFTER_TRAIN = true;//@ajay : changing from true;
  private static boolean CALL_TRAIN_FROM_DATUM_GEN = true; 

  private final String workDir;
  private final String serializedRelationExtractorName;
  private final String[] relationFeatures;
  private final int featureCountThreshold;
  private final KBPReader reader;
  private final FeatureFactory rff;
  private final RelationExtractorFactory factory;
  RelationExtractor relationExtractor;
  private final double samplingRatio;
  private Map<String, Set<String>> slotsByEntityId;
  
  /* Added by Ajay 28/09/2013*/
  private static Map<String, String> entitiesIdNameMap;

  public static Map<String, String> entitiesIdNameMap() {
	return entitiesIdNameMap;
}

public KBPTrainer(Properties props) throws Exception {
    this(props, false);
  }

  public KBPTrainer(Properties props, boolean trainOnly) throws Exception {
	
	entitiesIdNameMap = new HashMap<String, String>(); //Ajay: Initialization of the hashmap
	  
    Log.severe("Started constructor for KBPTrainer...");
    workDir = props.getProperty(Props.WORK_DIR);

    String srn = props.getProperty(Props.SERIALIZED_MODEL_PATH, "kbp_relation_model");
    if (srn.endsWith(Constants.SER_EXT))
      srn = srn.substring(0, srn.length() - Constants.SER_EXT.length());
    serializedRelationExtractorName = srn;
    // TODO: Give the serialised version of extractor a new name (in config. file) 
    Log.severe("serializedRelationExtractorName: " + serializedRelationExtractorName);
    relationFeatures = props.getProperty(Props.RELATION_FEATS).split(",\\s*");
    assert (relationFeatures != null && relationFeatures.length > 0);
    Log.severe("relationFeatures: " + StringUtils.join(relationFeatures));
    featureCountThreshold = PropertiesUtils.getInt(props, Props.FEATURE_COUNT_THRESHOLD, 5);
    Log.severe("featureCountThreshold: " + featureCountThreshold);
    boolean temporal = PropertiesUtils.getBool(props, Props.KBP_TEMPORAL);
    if(! trainOnly){
      reader = new KBPReader(props, false, temporal, false);
      reader.setLoggerLevel(Log.stringToLevel(props.getProperty(Props.READER_LOG_LEVEL)));
    } else {
      Log.severe("Will NOT construct a KBPReader object.");
      reader = null;
    }
    rff = new FeatureFactory(relationFeatures);
    // TODO: Do we need to comment the next line??  We need lexicalized arguments, I guess. But in this place it is of no use. 
    // To lexicalize the arguments, we need to regenerate the datum files ( addFeaturesRaw() )
    //rff.setDoNotLexicalizeFirstArgument(true);
    relationExtractor = null;
    factory = new RelationExtractorFactory(props.getProperty(Props.MODEL_TYPE, Constants.DEFAULT_MODEL));
    slotsByEntityId = readSlotsByEntityId(props);
    samplingRatio = PropertiesUtils.getDouble(props, 
        Props.NEGATIVES_SAMPLE_RATIO,
        Constants.DEFAULT_NEGATIVES_SAMPLING_RATIO);
    Log.severe("samplingRatio: " + samplingRatio);
    
    Log.severe("Completed constructor for KBPTrainer.");
  }

  private static Map<String, Set<String>> readSlotsByEntityId(Properties props) throws Exception {
    if (props.getProperty(Props.INPUT_KB) == null) {
      Log.severe("INPUT_KB not specified!!\n");
      return null;
    }
    KBPDomReader domReader = new KBPDomReader(props);
    Map<KBPEntity, List<KBPSlot>> entitySlots =
      domReader.parse(props.getProperty(Props.INPUT_KB));
    
    //Ajay;07/10/2013: Adding the entities for their names
    for(KBPEntity k : entitySlots.keySet()){
    	entitiesIdNameMap.put(k.id, k.name);
    }

    Map<String, Set<String>> slotsByEntityId = extractSlotsById(entitySlots);
    return slotsByEntityId;
  }

  public static Map<String, Set<String>> extractSlotsById(Map<KBPEntity, List<KBPSlot>> entitySlots) {
    Counter<String> slotStats = new ClassicCounter<String>();
    Map<String, Set<String>> slotsById = new HashMap<String, Set<String>>();
    for (KBPEntity ent : entitySlots.keySet()) {
      Set<String> mySlots = new HashSet<String>();
      Collection<KBPSlot> slots = entitySlots.get(ent);
      for (KBPSlot slot : slots) {
        mySlots.add(slot.slotName);
        slotStats.incrementCount(slot.slotName);
      }
       //System.err.println("SLOTS FOR ENTITY " + ent.id + ": " + mySlots);
      slotsById.put(ent.id, mySlots);
    }
    Log.severe("Slot stats in the KB: " + slotStats);
    return slotsById;
  }

  public boolean modelExists() {
    File modelFile = new File(makeModelPath());
    return FileSystem.existsAndNonEmpty(modelFile);
  }

  private String makeModelPath() {
    return makeModelPath(workDir, serializedRelationExtractorName, factory.modelType(), samplingRatio);
  }
  
  public static String makeModelPath(String workDir, 
      String serializedRelationExtractorName, 
      ModelType modelType, 
      double samplingRatio) {
    return workDir + File.separator + serializedRelationExtractorName + 
      "." + modelType + "." + (int) (100.0 * samplingRatio) +  
      Constants.SER_EXT;
  }

  public void setEntitySlotsById(Map<String, Set<String>> slots) {
    this.slotsByEntityId = slots;
  }

  public static void subsampleNegatives(List<File> trainDatumFiles, File negFile, double subsamplingProb) throws IOException {
    Log.severe("Subsampling negative datums...");
    PrintStream os = new PrintStream(new FileOutputStream(negFile));
    Random random = new Random(0);
    int total = 0, selected = 0;
    for (File trainDatumFile : trainDatumFiles) {
      BufferedReader is = new BufferedReader(new FileReader(trainDatumFile));
      for (String line; (line = is.readLine()) != null;) {
        List<MinimalDatum> minDatums = MinimalDatum.lineToDatum(line);
        for(MinimalDatum minDatum: minDatums){
          if(minDatum.datum().label().equals(RelationMention.UNRELATED)){
            total ++;
            if(random.nextDouble() < subsamplingProb){
              os.println(line);
              selected ++;
            }
          }
        }
      }
      is.close();
    }
    os.close();
    Log.severe("Extracted " + selected + " out of " + total + " negative examples in " + negFile.getAbsolutePath());
  }
  
  private static void generateNegativeLabels(Map<String, RelationDatum> datums,
      Map<String, Set<String>> slotsById) {
    for(String key: datums.keySet()) {
      RelationDatum datum = datums.get(key);
      
      List<String> knownSlots = new ArrayList<String>(slotsById.get(datum.entityId()));
      Collections.sort(knownSlots);

      // add this as negative example for all other known slots
      if (knownSlots != null && knownSlots.size() > 0) {
        // add this datum as negative example to all other labels from the known slots of this entity
        // Note: we consider ONLY these slot types because we don't know anything about other slots for this entity
        for (String otherSlot : knownSlots) {
          // skip my label(s); here we deal just with negative examples
          if (datum.yPos().contains(otherSlot)) continue;
          datum.addNeg(otherSlot);
        }
      }
    }
  }
  
  private static void loadRelationDatumsFromFile(File trainFile,
      Map<String, Set<String>> slotsById,
      boolean considerNegatives,
      Map<String, RelationDatum> datums) throws IOException {
    Log.severe("Processing file " + trainFile.getAbsolutePath() + "...");
    BufferedReader is = new BufferedReader(new FileReader(trainFile));
    int lineCount = 0;
    for (String line; (line = is.readLine()) != null;) {
      lineCount++;
      
      RelationDatum relDatum = RelationDatum.lineToDatum(line);
      if (slotsById.get(relDatum.entityId()) == null) {
        Log.fine("WARNING: Unknown slots for id: " + relDatum.entityId()
            + ". This happens because this entity was filtered out from the KB. Continuing.");
        continue;
      }
      
      if(relDatum.yPos().size() == 0 && ! considerNegatives){
        // skip negatives if considerNegatives is set
        continue;
      }
      
      // the current datums might have to be merged with a previously see one
      RelationDatum seenDatum = datums.get(relDatum.key());
      if(seenDatum == null){
        datums.put(relDatum.key(), relDatum);
      } else {
        seenDatum.merge(relDatum);
      }
      
      if (lineCount % 10000 == 0) {
        Log.severe("Loaded " + lineCount + " datums.");
      }
    }
    is.close();

    if(datums.size() == 0) {
      throw new RuntimeException("ERROR: cannot have 0 datums after loading a datum file!");
    }
  }
  
  private static Collection<RelationDatum> loadRelationDatums(
      List<File> trainDatumFiles, 
      File negFile,
      Map<String, Set<String>> slotsById) throws IOException {
    // map from (entity, slot) to the actual RelationDatum
    Map<String, RelationDatum> datums = new HashMap<String, RelationDatum>();
    
    for(File tf: trainDatumFiles){
      loadRelationDatumsFromFile(tf, slotsById, false, datums);
    }
    loadRelationDatumsFromFile(negFile, slotsById, true, datums);
    
    generateNegativeLabels(datums, slotsById);
    
    return datums.values();
  }
  
  public void trainAtLeastOnce(Properties props, List<File> trainDatumFiles, File negFile) throws IOException {
    assert(negFile != null);
    
    Collection<RelationDatum> datums = loadRelationDatums(trainDatumFiles, negFile, slotsByEntityId);
    
    //
    // some stats
    //
    for(RelationDatum datum: datums){
      if(datum.yPos().size() > 1) {
        //Log.severe("DATUM DUMP " + datum.entityType() + " " + datum.slotValue() + " (" + datum.datums().size() + "): " + datum.posAsString());
    	  Log.severe("DATUM DUMP " + datum.entityId() + " " + datum.entityType() + " " + datum.slotValue() + " (" + datum.datums().size() + "): " + datum.posAsString());
      }
    }
    Counter<String> posStats = new ClassicCounter<String>();
    Counter<String> negStats = new ClassicCounter<String>();
    Counter<Integer> sizeStats = new ClassicCounter<Integer>();
    for(RelationDatum datum: datums){
      posStats.incrementCount(datum.posAsString());
      negStats.incrementCount(datum.negAsString());
      sizeStats.incrementCount(datum.datums().size());
    }
    printStats(posStats, "POSITIVE");
    printStats(negStats, "NEGATIVE");
    Log.severe("Stats for datum size:");
    List<Pair<Integer,Double>> sorted = Counters.toDescendingMagnitudeSortedListWithCounts(sizeStats);
    for(Pair<Integer,Double> e: sorted) {
      Log.severe("DATUM SIZE " + e.first() + " seen " + e.second() + " times.");
    }
    
    //
    // build dataset and discard the datums
    //
    MultiLabelDataset<String, String> dataset = new MultiLabelDataset<String, String>();
    for(RelationDatum d: datums) {

    	//if(d.slotTypes().size() > 1)
//    	  	Log.fine("Arg2 of " + d.key() + " has types : " + d.slotTypes() + " for the slot value : " + d.slotValue());
    
//    	System.out.println("Key = " + d.key() + " -- " + entitiesIdNameMap().get(d.entityId()) + " : " + d.slotValue().replace('_', ' '));
    	
    	String entityVal = entitiesIdNameMap().get(d.entityId());
    	String slotVal = d.slotValue().replace('_', ' ');
    	dataset.addDatum(d.yPos(), d.yNeg(), d.datums(), entityVal, d.entityType(), slotVal, d.slotTypes());
//    	System.out.print(d.entityType()+"/");
//    	Set<String> slottype = new HashSet<String>();
//    	for(String s : d.slotTypes())
//    		slottype.add(s);
//    	System.out.print(slottype);
//    	System.out.print("/" + d.yPos());
//    	System.out.println();
    }
    datums = null; // can be GCed now
    //
    // feature selection
    //
    Log.severe("Applying feature selection with threshold " + featureCountThreshold + "...");
    dataset.applyFeatureCountThreshold(featureCountThreshold);
    
//    System.out.println("AJAY: DATASET SZ: "+ dataset.getDataArray().length);
//    int sz = 0;
//    int ndatums = 0;
//    for(int[][] datum : dataset.getDataArray()){
//    	ndatums++;
//    	sz += datum.length;
//    }
//    System.out.println("AJAY: DATASET SZ: "+ dataset.getDataArray().length + " AND " + ndatums );
//    System.out.println("AJAY: TOTAL NUMBER OF DATUMS: "+ sz);
    
    //
    // actual training
    //
    JointlyTrainedRelationExtractor extractor = factory.makeJointExtractor(props);
    extractor.train(dataset);
    extractor.save(makeModelPath());
    relationExtractor = extractor;
  }
  
  private static void printStats(Counter<String> stats, String name) {
    List<Pair<String,Double>> sorted = Counters.toDescendingMagnitudeSortedListWithCounts(stats);
    Log.severe("Stats for labels of type " + name);
    for(Pair<String,Double> l: sorted){
      Log.severe("LABEL: " + l.first() + " with count " + l.second());
    }
  }

  public void trainOneVsAll(List<File> trainDatumFiles, File negFile) throws IOException {
    // create one dataset for each label
    Map<String, GeneralDataset<String, String>> trainSets = null;
    
    assert(negFile != null);
    trainSets = createDatasetsWithOfflineNegatives(
        trainDatumFiles,
        negFile,
        slotsByEntityId);

    // feature selection
    for (String label : trainSets.keySet()) {
      GeneralDataset<String, String> corpus = trainSets.get(label);
      corpus.applyFeatureCountThreshold(featureCountThreshold);
    }

    // train one-vs-all classifiers
    OneVsAllRelationExtractor ovaRelationExtractor = new OneVsAllRelationExtractor();
    ovaRelationExtractor.train(trainSets);
    ovaRelationExtractor.save(makeModelPath());
    relationExtractor = ovaRelationExtractor;
  }

  static Set<String> allSlots = new HashSet<String>(Arrays.asList(
      "per:date_of_birth",
      "per:age",
      "per:country_of_birth",
      "per:stateorprovince_of_birth",
      "per:city_of_birth",
      "per:date_of_death",
      "per:country_of_death",
      "per:stateorprovince_of_death",
      "per:city_of_death",
      "per:cause_of_death",
      "per:religion",
      "org:number_of_employeesSLASHmembers",
      "org:founded",
      "org:dissolved",
      "org:country_of_headquarters",
      "org:stateorprovince_of_headquarters",
      "org:city_of_headquarters",
      "org:website",
      "per:alternate_names",
      "per:origin",
      "per:countries_of_residence",
      "per:stateorprovinces_of_residence",
      "per:cities_of_residence",
      "per:schools_attended",
      "per:title",
      "per:member_of",
      "per:employee_of",
      "per:spouse",
      "per:children",
      "per:parents",
      "per:siblings",
      "per:other_family",
      "per:charges",
      "org:alternate_names",
      "org:politicalSLASHreligious_affiliation",
      "org:top_membersSLASHemployees",
      "org:members",
      "org:member_of",
      "org:subsidiaries",
      "org:parents",
      "org:founded_by",
      "org:shareholders"));
  
  private static void loadFile(
      Map<String, GeneralDataset<String, String>> corporaByLabel,
      String fileName,
      Map<String, Set<String>> slotsById,
      boolean considerNegatives,
      boolean allNegatives,
      Counter<String> posStats,
      Counter<String> negStats) throws IOException {
    Log.severe("Processing file " + fileName);
    BufferedReader is = new BufferedReader(new FileReader(fileName));
    int lineCount = 0;
    for (String line; (line = is.readLine()) != null;) {
      lineCount++;
      
      List<MinimalDatum> minDatums = MinimalDatum.lineToDatum(line);
      for(MinimalDatum minDatum: minDatums) {
        if (slotsById.get(minDatum.entityId()) == null) {
          Log.fine("WARNING: Unknown slots for id: " + minDatum.entityId()
              + ". This happens because this entity was filtered out from the KB. Continuing.");
          continue;
        }

        BasicDatum<String, String> datum = (BasicDatum<String, String>) minDatum.datum();
        String label = datum.label();
        if(label.equals(RelationMention.UNRELATED) && ! considerNegatives) {
          continue;
        }

        List<String> knownSlots = (allNegatives ? 
            new ArrayList<String>(allSlots) :
              new ArrayList<String>(slotsById.get(minDatum.entityId())));
        Collections.sort(knownSlots);

        // store a positive example
        if(! label.equals(RelationMention.UNRELATED)){
          addDatum(corporaByLabel, label, datum, (float) 1.0);
          posStats.incrementCount(label);
        }

        // add this as negative example for all other known slots
        if (knownSlots != null && knownSlots.size() > 0) {
          // all negative examples have the label _NR
          BasicDatum<String, String> negDatum = new BasicDatum<String, String>(
              datum.asFeatures(),
              RelationMention.UNRELATED);

          // add this datum as negative example to all other labels from the known slots of this entity
          // Note: we consider ONLY these slot types because we don't know anything about other slots for this entity
          for (String otherSlot : knownSlots) {
            // skip my label; here we deal just with negative examples
            if (otherSlot.equals(label)) continue;

            addDatum(corporaByLabel, otherSlot, negDatum, (float) 1.0);
            negStats.incrementCount(otherSlot);
          }
        }
      }
      
      if (lineCount % 10000 == 0) {
        Log.severe("Loaded " + lineCount + " datums.");
      }
    }
    is.close();
  }
      
  
  private static Map<String, GeneralDataset<String, String>> createDatasetsWithOfflineNegatives(
      List<File> trainDatumFiles,
      File negFile,
      Map<String, Set<String>> slotsById) throws IOException {
    Map<String, GeneralDataset<String, String>> corporaByLabel = new HashMap<String, GeneralDataset<String, String>>();
    Counter<String> posStats = new ClassicCounter<String>();
    Counter<String> negStats = new ClassicCounter<String>();
    boolean allNegatives = false;

    //
    // Extract only positive examples from the regular datum files
    //
    int fileCount = 0;
    for (File trainDatumFile : trainDatumFiles) {
      loadFile(corporaByLabel, trainDatumFile.getAbsolutePath(), slotsById, false, allNegatives, posStats, negStats);
      fileCount ++;
      Log.severe("Processed " + fileCount + " out of " + trainDatumFiles.size() + " datum files.");
    }
    
    //
    // Extract negative examples from the dedicated file
    //
    Log.severe("Loading negative examples from offline file " + negFile);
    loadFile(corporaByLabel, negFile.getAbsolutePath(), slotsById, true, allNegatives, posStats, negStats);
    fileCount ++;
    Log.severe("Processed all " + fileCount + " datum files.");
    
    // keep only corpora where we have > 0 positive examples
    removeLabelsWithZeroExamples(corporaByLabel, posStats, negStats);
    
    return corporaByLabel;
  }

  private static void removeLabelsWithZeroExamples(
      Map<String, GeneralDataset<String, String>> corporaByLabel,
      Counter<String> posStats,
      Counter<String> negStats) {
    
    Set<String> labels = corporaByLabel.keySet();
    Set<String> toRemove = new HashSet<String>();
    for (String label : labels) {
      int posCount = (int) posStats.getCount(label);
      int negCount = (int) negStats.getCount(label);
      if (posCount == 0) {
        toRemove.add(label);
        Log.severe("Removed corpus for label " + label + " because it has 0 positive examples.");
      } else if (negCount == 0) {
        toRemove.add(label);
        Log.severe("Removed corpus for label " + label + " because it has 0 negative examples.");
      }
    }
    for (String label : toRemove) {
      corporaByLabel.remove(label);
    }
    
    Log.severe("DATASET LABEL STATS:");
    for (String label : labels) {
      int posCount = (int) posStats.getCount(label);
      int negCount = (int) negStats.getCount(label);
      int total = corporaByLabel.get(label).size();
      Log.severe("LABEL STATS for " + label + ": " + posCount + " (+) " + negCount + " (-) " + total + " (total)");
    }
  }

  private static void addDatum(Map<String, GeneralDataset<String, String>> corporaByLabel, String label,
      BasicDatum<String, String> datum, float datumWeight) {
    WeightedDataset<String, String> dataset = (WeightedDataset<String, String>) corporaByLabel.get(label);
    if (dataset == null) {
      dataset = new WeightedDataset<String, String>();
      corporaByLabel.put(label, dataset);
    }
    dataset.add(datum, datumWeight);
  }

  public void generateDatums(String kbPath, String datumFile) throws Exception {
    PrintStream os = new PrintStream(new FileOutputStream(datumFile));
    Counter<String> labelStats = new ClassicCounter<String>();
    Counter<String> domainStats = new ClassicCounter<String>();
    reader.parse(os, kbPath, rff, labelStats, domainStats);
    Log.severe("LABEL STATS: " + labelStats);
    Log.severe("DOMAIN STATS: " + domainStats);
    os.close();
  }

  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    Log.setLevel(Log.stringToLevel(props.getProperty(Props.LOG_LEVEL)));
    Log.severe("Using run id: " + props.getProperty(Props.RUN_ID) + " in working directory " + props.getProperty(Props.WORK_DIR));

    if (props.containsKey(Props.INPUT)) {
      datumGeneration(props);
    } else if(props.containsKey("negatives")) {
      generateNegatives(props);
    } else if (props.containsKey(Props.NLPSUB)) {
      if(Boolean.valueOf(props.getProperty(Props.NLPSUB))) {
        mapReduceMain(props, true);
      } else {
        mapReduceMain(props, false);
      }
    } else {
      // default: train followed by test
      // this is the former block enabled by Props.TRAINER
      train(props);
      if(EVALUATE_AFTER_TRAIN) evaluate(props);
    }
  }

  private static final String DATUM_GEN_MEMORY = "4g";
  private static final String TRAINER_MEMORY = "12g";
  static final String PROP_FILE = "kbp.properties";
  static final String KB_EXTENSION = ".xml";
  private static final int SLEEP_MILISECONDS = 60000;
  @SuppressWarnings("unused")
  private static final String HOSTS = "jude1,jude2,jude3,jude4,jude5,jude6,jude7,jude8,jude9";

  public static void mapReduceMain(Properties props, boolean inParallel) throws Exception {
    Log.severe("Running the map-reduce model...");
    File workDir = new File(props.getProperty(Props.WORK_DIR));
    assert (workDir.isDirectory());
    String trainPath = props.getProperty(Props.TRAIN_PATH);
    assert (trainPath != null);
    Log.severe("trainPath: " + trainPath);
    String testPath = props.getProperty(Props.TEST_PATH);
    assert (testPath != null);
    Log.severe("testPath: " + testPath);
    String scoreFile = workDir + File.separator + props.getProperty(Props.KB_SCORE_FILE);
    Log.severe("Scores will be saved in file: " + scoreFile);

    List<File> trainFiles = fetchFiles(trainPath, KB_EXTENSION);
    Log.severe("Found " + trainFiles.size() + " training files.");
    List<File> testFiles = fetchFiles(testPath, KB_EXTENSION);
    Log.severe("Found " + testFiles.size() + " testing files.");

    // save the properties to be used later by all slaves
    saveProperties(props, workDir);
    File trainDir = new File(workDir + File.separator + "train");
    @SuppressWarnings("unused")
    File testDir = new File(workDir + File.separator + "test");

    // create one datum generator job for each .XML file in training and testing
    for (File inputFile : trainFiles) {
      if (createDatumGenerationJob(inputFile, workDir, false, inParallel, props) == false) {
        throw new RuntimeException("Failed to create datum generation job for TRAINING file: "
            + inputFile.getAbsolutePath());
      }
    }
    if (TEST_ON_DEVEL) {
      for (File inputFile : testFiles) {
        if (createDatumGenerationJob(inputFile, workDir, true, inParallel, props) == false) {
          throw new RuntimeException("Failed to create datum generation job for TESTING file: "
              + inputFile.getAbsolutePath());
        }
      }
    }

    if (!inParallel) {
      train(props);
      
      // evaluate on actual KBP queries
      if(EVALUATE_AFTER_TRAIN) evaluate(props);
    } else if (!props.contains("notrain")){
      if(CALL_TRAIN_FROM_DATUM_GEN == false) {
        // wait until all datum generation jobs for training complete
        waitForChecks(trainDir, trainFiles.size());

        // everything finished; run the actual trainer PBS job
        if (createTrainerJob(workDir)) {
          Log.severe("Trainer job created successfully.");
        } else {
          throw new RuntimeException("ERROR: Failed to create trainer job!");
        }
      } else {
        Log.severe("All datum generation jobs created successfully.");
        Log.severe("The trainer job will be started when all datum generation jobs complete.");
      }
    }    
  }
  
  public static void evaluate(Properties props) throws Exception {
    // enable coref during testing!
    props.setProperty(Props.INDEX_PIPELINE_METHOD, "FULL");
    // we do not care about model combination mode here
    props.setProperty(Props.MODEL_COMBINATION_ENABLED, "false");
    KBPEvaluator.extractAndScore(props, false);
  }
  
  private static void generateNegatives(Properties props) throws Exception {
    File workDir = new File(props.getProperty(Props.WORK_DIR));
    assert (workDir.isDirectory());
    File trainDir = new File(workDir + File.separator + "train");
    List<File> trainDatumFiles = fetchFiles(trainDir.getAbsolutePath(), ".datums");
    List<Double> ratios = new ArrayList<Double>(Arrays.asList(1.00, 0.50, 0.25, 0.10));
    for(double ratio: ratios) {
      File negFile = new File(trainDir + File.separator + 
          "datums_" + (int) (100.0 * ratio) + ".negatives");
      if(! negFile.exists()) {
        KBPTrainer.subsampleNegatives(trainDatumFiles, negFile, ratio);
      }
    }
  }

  public static void train(Properties props) throws Exception {
    File workDir = new File(props.getProperty(Props.WORK_DIR));
    assert (workDir.isDirectory());
    File trainDir = new File(workDir + File.separator + "train");
    RelationExtractorFactory factory = 
      new RelationExtractorFactory(props.getProperty(Props.MODEL_TYPE, Constants.DEFAULT_MODEL));
    Log.severe("modelType = " + factory.modelType());
    double samplingRatio = PropertiesUtils.getDouble(props, 
        Props.NEGATIVES_SAMPLE_RATIO,
        Constants.DEFAULT_NEGATIVES_SAMPLING_RATIO);

    KBPTrainer trainer = new KBPTrainer(props, true);
    if (!trainer.modelExists()) {
      // construct dataset and train
      List<File> trainDatumFiles = fetchFiles(trainDir.getAbsolutePath(), ".datums");
      
      // generate a separate file with negative examples, so we have a fixed dataset for training
      File negFile = null;
      if(Constants.OFFLINE_NEGATIVES) {
        negFile = new File(trainDir + File.separator + 
            "datums_" + (int) (100.0 * samplingRatio) + ".negatives");
        if(! negFile.exists()) {
          KBPTrainer.subsampleNegatives(trainDatumFiles, negFile, samplingRatio);
        }
      }
      
      if (factory.isLocallyTrained()) {
        trainer.trainOneVsAll(trainDatumFiles, negFile);
      } else {
        assert(Constants.OFFLINE_NEGATIVES);
        trainer.trainAtLeastOnce(props, trainDatumFiles, negFile);
      } 
      Log.severe("Training complete.");
    }
  }

  static int countChecks(File workDir) throws InterruptedException {
    // sleep a few seconds, so NFS catches up
    Thread.sleep(3000);

    File[] checkFiles = workDir.listFiles(new FileFilter() {
      @Override
      public boolean accept(File arg0) {
        if (arg0.getAbsolutePath().endsWith(".check")) {
          return true;
        }
        return false;
      }
    });

    if (checkFiles != null)
      return checkFiles.length;
    return 0;
  }

  static boolean waitForChecks(File workDir, int howMany) throws InterruptedException {
    Log.severe("Waiting until we see " + howMany + " .check files in directory " + workDir + "...");
    int seconds = 0;
    while (true) {
      File[] checkFiles = workDir.listFiles(new FileFilter() {
        @Override
        public boolean accept(File arg0) {
          if (arg0.getAbsolutePath().endsWith(".check")) {
            return true;
          }
          return false;
        }
      });

      // we found the required number of .check files
      if (checkFiles != null && checkFiles.length >= howMany) {
        Log.severe("Found " + checkFiles.length + " .check files. Continuing execution.");
        break;
      }

      // sleep for one minute before the next check
      Thread.sleep(SLEEP_MILISECONDS);
      seconds += SLEEP_MILISECONDS / 1000;
      double minutes = seconds / 60;
      if (seconds % 600 == 0)
        Log.severe("Still waiting... " + minutes + " minutes have passed.");
    }
    return true;
  }

  private static void mkDir(File dir) throws InterruptedException {
    if (!dir.exists()) {
      dir.mkdirs();
    }
    System.err.println("Sleeping for a few seconds, to wait on NFS...");
    Thread.sleep(10000); // let NFS catch up
    System.err.println("Woke up.");
    assert (dir.exists() && dir.isDirectory());
  }

  private static boolean createTrainerJob(File workDir) throws Exception {
    String jobName = "KBPTrainer";
    File nlpsubDir = new File(workDir.getAbsolutePath() + File.separator + "nlpsub" + File.separator + "train");
    File logDir = new File(nlpsubDir.getAbsolutePath() + File.separator + jobName);
    mkDir(logDir);

    System.err.println("Creating KBPTrainer job: " + jobName);
    String cmd = "nlpsub -v --join-output-streams --queue=long --debug" +
        // " --priority=preemptable" +
        // " --hosts=" + HOSTS +
        // " --no-autodetect-memory" + // memory management SUCKS in PBS. do not use it
        " --cores=3"
        + // a single trainer job fits in a node, and we have 4 cores per node
        " --log-dir=" + logDir.getAbsolutePath() + " --name=" + jobName + " java -ea -Xmx" + TRAINER_MEMORY
        + " -XX:MaxPermSize=512m" + " edu.stanford.nlp.kbp.slotfilling.KBPTrainer" + " -props "
        + workDir.getAbsolutePath() + File.separator + PROP_FILE + " -" + Props.TRAINER + " true";

    boolean ret = launch(cmd);
    return ret;
  }

  private static boolean createDatumGenerationJob(File inputFile, File workDir, boolean isTest, boolean inParallel,
      Properties props) throws Exception {
    //
    // create the parent of all log directories
    //
    File nlpsubDir = null;
    if (inParallel) {
      nlpsubDir = new File(workDir.getAbsolutePath() + File.separator + "nlpsub" + File.separator
          + (isTest ? "test" : "train"));
      mkDir(nlpsubDir);
    }

    String jobName = getJobName(inputFile);

    // don't do anything if the output file has already been generated
    File datumDir = new File(workDir.getAbsolutePath() + File.separator + (isTest ? "test" : "train"));
    mkDir(datumDir);
    String outputFileName = datumDir.getAbsolutePath() + File.separator + jobName + ".datums";
    File outputFile = new File(outputFileName);
    
    if (FileSystem.existsAndNonEmpty(outputFile)) {
      System.err.println("Output file already exists: " + outputFileName);
      System.err.println("Skipping KBPDatumGenerator job: " + jobName);
      return true;
    }

    boolean ret = true;
    if (inParallel) {
      // create the actual log dir
      String logDirName = nlpsubDir.getAbsolutePath() + File.separator + jobName;
      File logDir = new File(logDirName);
      mkDir(logDir);

      String java = "java -ea -Xmx" + DATUM_GEN_MEMORY + " -XX:MaxPermSize=512m"
          + " edu.stanford.nlp.kbp.slotfilling.KBPTrainer" + " -props " + workDir.getAbsolutePath() + File.separator
          + PROP_FILE + " -" + Props.INPUT + " " + inputFile.getAbsolutePath() + " -" + Props.OUTPUT + " "
          + outputFileName + " -" + Props.IS_TEST + " " + isTest;

      System.err.println("Creating KBPDatumGenerator job: " + jobName);
      String cmd = "nlpsub -v --join-output-streams --queue=long --debug" +
          // " --priority=preemptable" +
          // " --hosts=" + HOSTS +
          // " --no-autodetect-memory" + // memory management SUCKS in PBS. do not use it
          " --cores=2" + // about two datum generation jobs fit in a node, and we have 4 cores per node
          " --log-dir=" + logDirName + " --name=" + jobName + " " + java;

      ret = launch(cmd);

      System.err.println("If you want to run this job manually use this command:\n" + "nohup " + java + " >& "
          + logDirName + File.separator + "nohup.out &");
    } else {
      KBPTrainer trainer = new KBPTrainer(props);
      trainer.generateDatums(inputFile.getAbsolutePath(), outputFileName);
    }

    return ret;
  }

  private static boolean launch(String cmd) throws IOException, InterruptedException {
    System.err.println("Launching command: " + cmd);

    ProcessWrapper process = ProcessWrapper.create(cmd);
    process.waitFor();
    int exitCode = process.exitValue();
    if (exitCode == 0) {
      System.err.println("The above command terminated successfully.");
    } else {
      System.err.println("The above command exited with code: " + exitCode);
    }

    String err = process.consumeErrorStream();
    if (err.length() > 0) {
      System.err.println("Error stream contained:\n" + err);
    }
    String stdout = process.consumeReadStream();
    if (stdout.length() > 0) {
      System.err.println("Stdout stream contained:\n" + stdout);
    }

    return (exitCode == 0);
  }

  static String getJobName(File inputFile) {
    int lastSlashPos = inputFile.getAbsolutePath().lastIndexOf((int) File.separatorChar);
    int extStart = inputFile.getAbsolutePath().lastIndexOf(KB_EXTENSION);
    assert (extStart > lastSlashPos);
    String jobName = inputFile.getAbsolutePath().substring(lastSlashPos + 1, extStart);
    return jobName;
  }

  private static void saveProperties(Properties props, File workDir) throws IOException {
    PrintStream os = new PrintStream(new FileOutputStream(workDir.getAbsolutePath() + File.separator + PROP_FILE));
    List<String> keys = new ArrayList<String>(props.stringPropertyNames());
    Collections.sort(keys);
    for (Object key : keys) {
      os.println(key.toString() + " = " + props.get(key).toString());
    }
    os.close();
  }

  static public List<File> fetchFiles(String path, final String extension) {
    File kbDir = new File(path);
    assert (kbDir.isDirectory());
    File[] inputFiles = kbDir.listFiles(new FileFilter() {
      @Override
      public boolean accept(File pathname) {
        if (pathname.getAbsolutePath().endsWith(extension))
          return true;
        return false;
      }
    });
    List<File> files = Arrays.asList(inputFiles);
    Collections.sort(files, new Comparator<File>() {
      @Override
      public int compare(File o1, File o2) {
        return o1.getAbsolutePath().compareTo(o2.getAbsolutePath());
      }
    });
    return files;
  }
  
  private static void datumGeneration(Properties props) throws Exception {
    String kbPath = props.getProperty(Props.INPUT);
    String datumFile = props.getProperty(Props.OUTPUT);
    @SuppressWarnings("unused")
    boolean isTest = PropertiesUtils.getBool(props, Props.IS_TEST);
    KBPTrainer trainer = new KBPTrainer(props);
    trainer.generateDatums(kbPath, datumFile);

    PrintStream os = new PrintStream(new FileOutputStream(datumFile + ".check"));
    os.println("Ok");
    os.close();
    Log.severe("Datums successfully saved in " + datumFile);
    Log.severe("Check file saved as " + datumFile + ".check");

    if(CALL_TRAIN_FROM_DATUM_GEN){
      String trainPath = props.getProperty(Props.TRAIN_PATH); 
      assert (trainPath != null); 
      Log.severe("trainPath: " + trainPath); 
      List<File> trainFiles = fetchFiles(trainPath, KB_EXTENSION);
      int checkCount = countChecks(new File(trainer.workDir + File.separator + "train")); 
      if(checkCount == trainFiles.size()) { 
        Log.severe("Found all " + checkCount + " datum files. Starting trainer job..."); // everything finished; run the actual trainer PBS job 
        boolean ret = createTrainerJob(new File(trainer.workDir));
        if(ret) Log.severe("Trainer job created successfully.");
        else Log.severe("ERROR: Failed to create trainer job!");
      } else { 
        Log.severe("Found only " + checkCount + " out of " + trainFiles.size() +
        " datum files. Will NOT start trainer job."); 
      } 
    }
  }
}
