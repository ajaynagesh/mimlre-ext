/**
 * Verifies how many correct answers exists in the candidates proposed *after* each stage, i.e.: IR, NE identification, NE classification
 * This class essentially copies the first part of KBPEvaluator (IR + NER), so it is a more realistic experiment
 * Note: this class used to be ErrorAnalysisAfterNer in the kbp package
 */
package edu.stanford.nlp.kbp.slotfilling;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.kbp.slotfilling.common.GoldResponses;
import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.kbp.slotfilling.common.KBPSlot;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.kbp.slotfilling.common.SlotToNamedEntities;
import edu.stanford.nlp.kbp.slotfilling.common.SlotsToNamedEntities;
import edu.stanford.nlp.kbp.slotfilling.common.StringFinder;
import edu.stanford.nlp.kbp.slotfilling.common.Utils;
import edu.stanford.nlp.kbp.slotfilling.distantsupervision.KBPReader;
import edu.stanford.nlp.kbp.slotfilling.index.PipelineIndexExtractor;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.StringUtils;

public class StageErrorAnalysis {
  private KBPReader reader;
  private SlotsToNamedEntities NEREntries;
  Map<String, Set<String>> neToSlots;
  
  GoldResponses goldResponses;
  
  Counter<String> totalPerSlot;
  Counter<String> correctPerSlotAfterNer;
  Counter<String> correctPerSlotAfterNerWithClass;
  Counter<String> correctPerSlotBeforeNer;
  
  final boolean testMode;
  final boolean temporal;
  final boolean diagnosticMode;
  public StageErrorAnalysis(Properties props) throws Exception {
    testMode = Boolean.valueOf(props.getProperty(Props.ANALYSIS_TEST_MODE, "true"));
    temporal = PropertiesUtils.getBool(props, Props.KBP_TEMPORAL, false);
    diagnosticMode = PropertiesUtils.getBool(props, Props.KBP_DIAGNOSTICMODE, false);
    reader = new KBPReader(props, testMode, false, false);
    reader.setLoggerLevel(Log.stringToLevel(props.getProperty(Props.READER_LOG_LEVEL, "INFO")));
    NEREntries = new SlotsToNamedEntities(props.getProperty(Props.NERENTRY_FILE));
    neToSlots = NEREntries.buildNeToSlots();
    goldResponses = new GoldResponses(props.getProperty(Props.GOLD_RESPONSES));
    totalPerSlot = new ClassicCounter<String>();
    correctPerSlotAfterNer = new ClassicCounter<String>();
    correctPerSlotAfterNerWithClass = new ClassicCounter<String>();
    correctPerSlotBeforeNer = new ClassicCounter<String>();
  }
  
  public boolean testMode() { return testMode; }
  
  private void analyze(Map<KBPEntity, List<KBPSlot>> entitySlotValues, Properties props) throws Exception {
    List<KBPEntity> sortedEntities = new ArrayList<KBPEntity>(entitySlotValues.keySet());
    Collections.sort(sortedEntities, new Comparator<KBPEntity>() {
      public int compare(KBPEntity o1, KBPEntity o2) {
        return o1.name.compareTo(o2.name);
      }
    });

    FileOutputStream fout = null;
    ObjectOutputStream oout = null;
    Map<String, List<CoreMap>> savedSentences = 
      new HashMap<String, List<CoreMap>>();
    String dumpFilename = props.getProperty("analysis.dumpSentences");
    if (dumpFilename != null) {
      fout = new FileOutputStream(dumpFilename);
      oout = new ObjectOutputStream(fout);
      System.err.println("Dumping sentences to file " + dumpFilename);
    }
    // extract sentences for each individual entity
    for(KBPEntity entity: sortedEntities){
      analyzeEntity(entity, entitySlotValues, savedSentences);
    }

    if (oout != null) {
      oout.writeObject(savedSentences);
      oout.close();
    }
    if (fout != null) {
      fout.close();
    }
  }
  
  private static String subtractEntity(String text, List<Pair<Integer, Integer>> matches) {
    StringBuffer os = new StringBuffer();
    int end = 0;
    if(matches != null){
      for(Pair<Integer, Integer> match: matches){
        String sub = text.substring(end, match.first());
        os.append(sub);
        end = match.second();
      }
    }
    os.append(text.substring(end));
    return os.toString();
  }
  
  private boolean compatible(String slot, String ne) {
    SlotToNamedEntities entry = NEREntries.getSlotInfo(slot);
    if(entry != null && entry.validNamedEntityLabels().contains(ne)) return true;
    return false;
  }
  
  private List<Set<String>> getGoldResponses(KBPEntity entity, List<KBPSlot> knownSlots, String slotName) {
    if(testMode) {
      return goldResponses.getResponses(entity.queryId, slotName);
    } else {
      if(knownSlots == null) return null;
      List<Set<String>> slotsAsStrings = new ArrayList<Set<String>>();
      for(KBPSlot knownSlot: knownSlots) {
        if(knownSlot.slotName.equalsIgnoreCase(slotName)){
          Set<String> queries = new HashSet<String>();
          queries.addAll(PipelineIndexExtractor.slotToQueries(knownSlot, false));
          slotsAsStrings.add(queries);
        }
      }
      if(slotsAsStrings.size() == 0) return null;
      return slotsAsStrings;
    }
  }
  
  private List<Pair<Integer, Integer>> removeEntityOverlap(List<Pair<Integer, Integer>> slotOffsets, List<Pair<Integer, Integer>> entOffsets) {
    List<Pair<Integer, Integer>> cleanOffsets = new ArrayList<Pair<Integer,Integer>>();
    for(Pair<Integer, Integer> slotOffset: slotOffsets) {
      boolean found = false;
      for(Pair<Integer, Integer> entOffset: entOffsets) {
        if(entOffset.first() <= slotOffset.first() && slotOffset.second() <= entOffset.second()) {
          found = true;
          break;
        }
      }
      if(! found) cleanOffsets.add(slotOffset);
    }
    return cleanOffsets;
  }
  
  /**
   * Pretty display of entity/slot/sentence
   * This format is used by SemgrexTester to evaluate patterns
   * @param os
   * @param sentence
   * @param prettyText
   * @param slotName
   * @param entOffsets
   * @param slotOffsets
   */
  private static void displaySentenceWithMatches(PrintStream os, 
      String sentence,
      String prettyText,
      String slotName, 
      List<Pair<Integer, Integer>> entOffsets, 
      List<Pair<Integer, Integer>> slotOffsets) {
    // TODO: not sure why this happens, but we might have entOffsets.size() == 0. Coref?
    if(entOffsets.size() == 0 || slotOffsets.size() == 0) return;
    
    Set<String> ents = new HashSet<String>();
    for(int i = 0; i < entOffsets.size(); i ++){
      ents.add(sentence.substring(entOffsets.get(i).first(), entOffsets.get(i).second()));
    }
    
    Set<String> slots = new HashSet<String>();
    for(int i = 0; i < slotOffsets.size(); i ++){
      slots.add(sentence.substring(slotOffsets.get(i).first(), slotOffsets.get(i).second()));
    }
    
    os.print("MATCHED " + slotName);
    os.print("\t||");
    for(String ent: ents) os.print("\t" + ent);
    os.print("\t||");
    for(String slot: slots) os.print("\t" + slot);
    os.print("\t||\t" + prettyText);
  }
  
  private boolean analyzeEntity(KBPEntity entity, Map<KBPEntity, List<KBPSlot>> entitySlotValues, Map<String, List<CoreMap>> savedSentences) throws Exception {
    // Extract all sentences that contain mentions of the entities in the test queries
    List<CoreMap> sentences = reader.readEntity(entity, entitySlotValues, null, null, true);
    StringFinder entFinder = new StringFinder(entity.name);
    String eid = entity.queryId;
    
    int myTotal = 0, myCorrectBefore = 0, myCorrectAfter = 0, myCorrectAfterWithClass = 0;
    for(String slotName : NEREntries.keySet()) {
      List<Set<String>> acceptableAnswers = getGoldResponses(entity, entitySlotValues.get(entity), slotName);
      if (acceptableAnswers == null) {
        Log.fine("Skipping " + slotName + " for " + eid);
        continue;
      }
      
      Log.severe("GOLD slots for entity " + entity.name + " and slot " + slotName + ": " + acceptableAnswers);
      List<StringFinder> finders = new ArrayList<StringFinder>();
      for (Set<String> answers : acceptableAnswers) {
        finders.add(new StringFinder(answers, true));
      }
      
      myTotal += finders.size();
      totalPerSlot.incrementCount("all", finders.size());
      totalPerSlot.incrementCount(slotName, finders.size());

      // check all gold responses for this slot type (slotName)
      for(StringFinder finder: finders){    
        boolean countedBefore = false, countedAfter = false, countedAfterWithClass = false;
        for(CoreMap sentence: sentences){
          boolean foundBefore = false, foundAfter = false, foundAfterWithClass = false;
          String fullText = StringFinder.toMatchString(sentence);
          String prettyText = Utils.sentenceToMinimalString(sentence);
          List<Pair<Integer, Integer>> entOffsets = entFinder.whereItMatches(fullText);
          String textWithoutEntity = subtractEntity(fullText, entOffsets);
          Log.severe("Original sentence: " + fullText);
          Log.severe("Sentence without entity matches: " + textWithoutEntity);
          List<Pair<Integer, Integer>> slotOffsets = null;

          // found before NER
          if(finder.matches(textWithoutEntity)){
            foundBefore = true;
            if(! countedBefore){
              correctPerSlotBeforeNer.incrementCount("all");
              correctPerSlotBeforeNer.incrementCount(slotName);
              myCorrectBefore ++;
              countedBefore = true;
            }
            
            slotOffsets = finder.whereItMatches(fullText);
            slotOffsets = removeEntityOverlap(slotOffsets, entOffsets);
          }
          
          // found after NER
          List<RelationMention> relationMentions = sentence.get(MachineReadingAnnotations.RelationMentionsAnnotation.class);
          if(relationMentions != null){
            for(RelationMention rel: relationMentions){
              String value = rel.getArg(1).getExtentString();
              if(finder.matches(value)){
                foundAfter = true;
                if(! countedAfter){
                  correctPerSlotAfterNer.incrementCount("all");
                  correctPerSlotAfterNer.incrementCount(slotName);
                  myCorrectAfter ++;
                  countedAfter = true;
                }
                break;
              }
            }
          }
          
          if(relationMentions != null){
            for(RelationMention rel: relationMentions){
              String value = rel.getArg(1).getExtentString();
              if(finder.matches(value) && compatible(slotName, rel.getArg(1).getType())){
                foundAfterWithClass = true;
                if(! countedAfterWithClass){
                  correctPerSlotAfterNerWithClass.incrementCount("all");
                  correctPerSlotAfterNerWithClass.incrementCount(slotName);
                  myCorrectAfterWithClass ++;
                  countedAfterWithClass = true;
                }
                break;
              }
            }
          }
          
          if(slotOffsets != null && slotOffsets.size() > 0 && foundAfterWithClass){
            displaySentenceWithMatches(System.err, fullText, prettyText, slotName, entOffsets, slotOffsets);
            if (savedSentences != null) {
              List<CoreMap> sentenceList = savedSentences.get(slotName);
              if (sentenceList == null) {
                sentenceList = new ArrayList<CoreMap>();
                savedSentences.put(slotName, sentenceList);
              }
              savedSentences.get(slotName).add(sentence);
            }
          }
          
          if(foundBefore == true && foundAfter == false){
            Log.severe("For entity " + eid + " and slot " + slotName + " with value " + finder + ", did not find anything after NER in sentence:\n" + Utils.sentenceToString(sentence));
          }
          if(foundAfter == true && foundAfterWithClass == false){
            Log.severe("For entity " + eid + " and slot " + slotName + " with value " + finder + ", did not match NE labels in sentence:\n" + Utils.sentenceToString(sentence));
          }
        }
      }
    }
      
    Log.severe("Matched " + myCorrectBefore  + " (before NER); " + 
        myCorrectAfter + " (after NER); and " + 
        myCorrectAfterWithClass + " (after NER with type match) out of " + 
        myTotal + " GOLD slots for entity " + eid);    
    return true;
  }
  
  public void score(PrintStream os) {
    List<String> slotTypes = new ArrayList<String>(totalPerSlot.keySet());
    Collections.sort(slotTypes);
    slotTypes.remove(0); // "all" appear first but we want it at the end
    slotTypes.add("all");
    os.println("Slot name\tCorrect Before NER\tCorrect After NER\tCorrect After NER with Class\tTotal\tRecall Before NER\tRecall After NER\tRecall After NER with Class");
    for(String slotType: slotTypes) {
      double ca = correctPerSlotAfterNer.getCount(slotType);
      double cac = correctPerSlotAfterNerWithClass.getCount(slotType);
      double cb = correctPerSlotBeforeNer.getCount(slotType);
      double t = totalPerSlot.getCount(slotType);
      double recallAfter = 100 * ca / t;
      double recallAfterWithClass = 100 * cac / t;
      double recallBefore = 100 * cb / t;
      os.println(slotType + "\t" + cb + "\t" + ca + "\t" + cac + "\t" + t + "\t" + recallBefore + "\t" + recallAfter + "\t" + recallAfterWithClass);
    }
  }
  
  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    Log.setLevel(Log.stringToLevel(props.getProperty(Props.LOG_LEVEL, "INFO")));
    System.err.println("Using the following properties: " + props);
    
    // convert path names to use the local machine name
    /*
    props.setProperty(Props.INDEX_PROP, KBPDatumGenerator.convertToHostName(props.getProperty(Props.INDEX_PROP)));
    Log.severe("Actual " + Props.INDEX_PROP + " property used: " + props.getProperty(Props.INDEX_PROP));
    props.setProperty(Props.WEB_CACHE_PROP, KBPDatumGenerator.convertToHostName(props.getProperty(Props.WEB_CACHE_PROP)));
    Log.severe("Actual " + Props.WEB_CACHE_PROP + " property used: " + props.getProperty(Props.WEB_CACHE_PROP));
    props.setProperty(Props.OFFICIAL_INDEX_PROP, KBPDatumGenerator.convertToHostName(props.getProperty(Props.OFFICIAL_INDEX_PROP)));
    Log.severe("Actual " + Props.OFFICIAL_INDEX_PROP + " property used: " + props.getProperty(Props.OFFICIAL_INDEX_PROP));
    */
    
    StageErrorAnalysis script = new StageErrorAnalysis(props);
    
    Map<KBPEntity, List<KBPSlot>> entitySlotValues = 
      (script.testMode() ? 
          script.reader.loadEntitiesAndSlots(props.getProperty(Props.DEV_QUERIES)) :
          script.reader.loadEntitiesAndSlots(props.getProperty(Props.ANALYSIS_KB)));
    
    script.analyze(entitySlotValues, props);
    System.out.println("Retrieved " + 
        (script.testMode() ? props.getProperty(Props.TEST_SENTENCES_PER_ENTITY) : props.getProperty(Props.TRAIN_SENTENCES_PER_ENTITY)) +
        " sentences per entity.");
    script.score(System.out);
  }
}
