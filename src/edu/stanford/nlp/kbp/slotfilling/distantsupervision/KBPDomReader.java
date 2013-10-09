package edu.stanford.nlp.kbp.slotfilling.distantsupervision;

import java.io.BufferedReader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import java.util.regex.Pattern;
import java.util.regex.Matcher;

import javax.xml.parsers.ParserConfigurationException;

import edu.stanford.nlp.kbp.slotfilling.common.*;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Timing;

/**
 * Dom reader for the KBP specification. Given a directory of documents containing Wikipedia infobox
 * entries, parses each document to extract standard KBP relations and outputs them as a list of 
 * RelationMentions.
 */

public class KBPDomReader extends DomReader {
  private final String MAP_DIR; 
  private final String NER_TYPES; 
  private final String MANUAL_LISTS; 
  private final String COUNTRIES; 
  private final String STATES_AND_PROVINCES; 
	
  private Map<String, Integer> missedEntities = new HashMap<String, Integer>();
  private Set<KBPSlot> relationMentions = new HashSet<KBPSlot>();
	
  /** because one-to-many mappings are possible, the value type is a set of relation names */
  private final Map<String, Set<String>> infoboxToKBP;
	
  /** keeps track of the possible infobox entity classes, that is those which 
     correspond to KBP people or organizations */
  private Set<String> entityClasses = new HashSet<String>();
  private Set<String> perEntityClasses = new HashSet<String>();
  private Set<String> orgEntityClasses = new HashSet<String>();
	
  /** holds the NER type that each relation expects, or if there is no associated NER type,
     the set of possible fillers for this relation */
  private Map<String, Set<String>> NERTypes = new HashMap<String, Set<String>>();
  
  private final Set<String> countries = new HashSet<String>();
  private final Set<String> statesAndProvinces = new HashSet<String>();
  
  /** Map from nationalities to country names */
  private final Nationalities nationalities;
  
  /** Map from slot names to accepted NE fillers */ 
  private final SlotsToNamedEntities slotsToNamedEntities;
  // TODO: the above is possibly redundant with NERTypes. Check and remove NERTypes 
    
  public KBPDomReader(Properties props) throws IOException {
    infoboxToKBP = new HashMap<String, Set<String>>();
    NERTypes = new HashMap<String, Set<String>>();
    entityClasses = new HashSet<String>();
    perEntityClasses = new HashSet<String>();
    orgEntityClasses = new HashSet<String>();
    slotsToNamedEntities = new SlotsToNamedEntities(props.getProperty(Props.NERENTRY_FILE, "/u/nlp/data/TAC-KBP2010/sentence_extraction/NER_types"));
    
    MAP_DIR = props.getProperty(Props.KBP_MAPPING, "/u/nlp/data/TAC-KBP2010/clean_knowledge_base/docs/mapping");
    NER_TYPES = props.getProperty(Props.NERENTRY_FILE, "/u/nlp/data/TAC-KBP2010/clean_knowledge_base/docs/NER_types");
    MANUAL_LISTS = props.getProperty(Props.KBP_MANUAL_LISTS, "/u/nlp/data/TAC-KBP2010/clean_knowledge_base/docs/manual_lists/specific_relations");
    COUNTRIES = props.getProperty(Props.KBP_COUNTRIES, "/u/nlp/data/TAC-KBP2010/clean_knowledge_base/docs/manual_lists/locations/countries");
    STATES_AND_PROVINCES = props.getProperty(Props.KBP_STATES, "/u/nlp/data/TAC-KBP2010/clean_knowledge_base/docs/manual_lists/locations/statesandprovinces");
    
    readMapFiles();
    readNERTypes();
    readLocations();
    
    if(props.containsKey(Props.NATIONALITIES)) {
      nationalities = new Nationalities(props.getProperty(Props.NATIONALITIES));
    } else {
      nationalities = null;
    }
  }
  
  public Nationalities nationalities() { return nationalities; }
  
  /**
   * Reads either a directory containing infobox files (*.xml) or a single file.  Returns a map between {@link KBPEntity} and the {@link KBPSlot}s
   * they're associated with. 
   */
  public Map<KBPEntity, List<KBPSlot>> parse(String kbPath) throws IOException, SAXException, ParserConfigurationException {
    missedEntities = new HashMap<String, Integer>();
    relationMentions = new HashSet<KBPSlot>();
    
    File [] files = null;
    File dir = new File(kbPath);
    if(dir.isFile()){
      files = new File[1];
      files[0] = dir;
    } else {
      files = dir.listFiles();
    }
    
    System.err.println("Loading KB from " + kbPath + "...");
    Timing tim = new Timing();
    Timing.startTime();
    for (int i = 0; i < files.length; i++) {
      if(files[i].getAbsolutePath().endsWith(".xml")){
        // System.out.println("parsing document... " + i);
        parseDocument(files[i]);
      }
    }
    System.err.println("Completed loading the KB [" + tim.toSecondsString() + "]");
    
    return getRelationsMap();
  }

  private void parseDocument(File file)
    throws IOException, SAXException, ParserConfigurationException {
    Document document = readDocument(file);
    
    NodeList entities = document.getElementsByTagName("entity");
    for (int i = 0; i < entities.getLength(); i++){
      Set<KBPSlot> rels = parseEntity(entities.item(i));
      for(KBPSlot rel: rels) {
        // add alternate names to slots
        rel.addAlternateSlotValues(slotsToNamedEntities, nationalities);
      }

      // store all the new relations for future use
      relationMentions.addAll(rels);
    }
  }
  
  private String findBackground(Node facts) {
    List<Node> children = getChildrenByName(facts, "fact");
    for(Node child: children) {
      String name = getAttributeValue(child, "name");
      if(name.equalsIgnoreCase("background")){
        return getText(child);
      }
    }
    return null;
  }
  
  private static final Set<String> BANDS = new HashSet<String>(Arrays.asList("group_or_band", "classical_ensemble"));
  
  private Set<KBPSlot> parseEntity(Node entity) {
    Set<KBPSlot> result = new HashSet<KBPSlot>();
    String entityName = getAttributeValue(entity, "name");
    entityName = removeParentheses(entityName).trim();
    String entityId = getAttributeValue(entity, "id");
    if(entityId == null) throw new RuntimeException("Unknown id for entity " + entityName);
  	
    Node facts = getChildByName(entity, "facts");
    String entityClass = getAttributeValue(facts, "class");
    String background = findBackground(facts);
    // if(background != null) System.err.println("Found background: " + background);
    EntityType entType = null; 
    if(perEntityClasses.contains(entityClass.toLowerCase())){
      if(background != null && BANDS != null && BANDS.contains(background.toLowerCase())){
        Log.fine("Found group_or_band: " + entityName);
      } else {
        entType = EntityType.PERSON;
      } 
    }
    else {
      if(orgEntityClasses.contains(entityClass.toLowerCase())) {
        entType = EntityType.ORGANIZATION;
      }
    }
  	
    /* ensure that this entity corresponds to a KBP person or organization */
    if (entType == null){
      Integer value = missedEntities.get(entityClass);
      if (value == null) missedEntities.put(entityClass, 1);
      else missedEntities.put(entityClass, value + 1);
      return result;
    }
    
    // we get here only with PER or ORG
    assert(entType != null);
 
    List<Node> fillers = getChildrenByName(facts, "fact");
    for (Node filler : fillers) {
      String relationName = getAttributeValue(filler, "name");
  		
      /* ensure that this relation a recognized KBP relation */
      Set<String> relations = infoboxToKBP.get(entityClass + ":" + relationName);
      if (relations == null) continue; 
  		
      result.addAll(extractAllRelations(entityName, entityId, filler, relations));
    }
    
    // set entity type in all these relations
    for(KBPSlot rel: result){
      rel.entityType = entType;
    }
    
    return result;
  }
  
  private Set<KBPSlot> extractAllRelations(String entityName, String entityId, Node filler, Set<String> relations) {
    StringBuffer os = new StringBuffer();
    os.append(entityName + "\t" + relations + "\t" + getTextWithLinks(filler));
    
    Set<KBPSlot> result = new HashSet<KBPSlot>();
    
    if (relations.size() == 1) {  /* the relation is unambiguous */
      String relationName = relations.iterator().next();
      List<Node> links = getChildrenByName(filler, "link");
			
      if (links.size() > 0 && !NERTypes.get(relationName).contains("DATE")) {   
        for (Node link : links) { /* add the text of each link as a separate RelationMention */
          if (link.getChildNodes().getLength() > 0) {
            String fillerName = removeParentheses(getText(link)).trim();
            result.addAll(extractUnambiguousRelation(entityName, entityId, fillerName, relationName));
          }
        }
      } else { /* the filler does not contain links, or is a date relation */
        String text = removeParentheses(getTextWOLinks(filler));
        String[] fillerNames = {text};
        if (!NERTypes.get(relationName).contains("DATE") && !NERTypes.get(relationName).contains("NUMBER")) 
          fillerNames = text.split(",(?!\\s+(Jr|Sr|I|II|III|IV)\\.*)\\s+|\\s+and\\s+|\\s+or\\s+|\\n|\\t"); 
       
        for (String fillerName : fillerNames) {
          fillerName = fillerName.trim();
          result.addAll(extractUnambiguousRelation(entityName, entityId, fillerName, relationName));
        }
      }
    } else { /* relation is ambiguous, we must parse the proposed filler */
      String fillerName = getTextWOLinks(filler);
      /* handle the age relation specially, then remove parentheses */
      Pattern pattern = Pattern.compile("\\((age|aged)\\W+[0-9]+\\)");
      Matcher matcher = pattern.matcher(fillerName);
      if (matcher.find())
        result.addAll(extractUnambiguousRelation(entityName, entityId, matcher.group(), "per:age"));
      fillerName = removeParentheses(fillerName);
      result.addAll(extractAmbiguousRelations(entityName, entityId, fillerName, relations));
    }
    
    for (KBPSlot mention : result)
      os.append(", " + mention.slotName + "\t" + mention.slotValue);
    Log.fine(os.toString());
    
    return result;
  }
  
  /**
   * Given the relation type and a proposed filler, cleans the filler and extracts a RelationMention object
   * then adds the object to the global relationMentions map. Location relations are handled separately,
   * through the extractLocationRelation method defined below.
   */
  private Set<KBPSlot> extractUnambiguousRelation(String entityName, String entityId, String fillerName, String relationName) {
    Set<KBPSlot> result = new HashSet<KBPSlot>();
    
    if (fillerName.equals(entityName)) return result;
  	
    Set<String> NERtype = NERTypes.get(relationName);
    String regex = null;

    if (relationName.equals("org:founded")) {
      if (fillerName.matches("[^-]+\\W*-\\W*[^-]+"))
        fillerName = fillerName.split("\\W*-\\W*")[0]; 
    } else if (relationName.equals("org:dissolved")) {
      if (fillerName.matches("[^-]+\\W*-\\W*[^-]+"))
        fillerName = fillerName.split("\\W*-\\W*")[1]; 
    }
    
    if (NERtype.contains("DATE")) regex = "[A-Za-z0-9, -/]*[0-9]";
    else if (NERtype.contains("NUMBER")) regex = "[0-9]+";
    else if (NERtype.contains("URL")) regex = ".*www\\..*(\\.com|\\.edu|\\.org|\\.uk|\\.us|\\.net)\\..*";
    else if (NERtype.contains("RELIGION")) regex = "[A-Z].*";
    else if (NERtype.contains("LOCATION")) regex = "[A-Z].*[A-Za-z]";
    else if (NERtype.contains("PERSON")) regex = "[A-Z][A-Za-z ,\\.]*[a-z\\.]";
    else if (NERtype.contains("ORGANIZATION")) {
      if (fillerName.equalsIgnoreCase("public") || fillerName.equalsIgnoreCase("private")) return result;
      regex = "[A-Z].*";
    }
      
    if (regex != null) { /* the relation expects a certain NER type */
      Pattern pattern = Pattern.compile(regex);
      Matcher matcher = pattern.matcher(fillerName);
      if (matcher.find())
        result.add(new KBPSlot(entityName, entityId, matcher.group(), relationName));
    } else if (NERTypes.get(relationName).contains(fillerName.toLowerCase())) { // otherwise, use the manual lists
      result.add(new KBPSlot(entityName, entityId, fillerName, relationName));
    }
    return result;
  }
  
  /**
   *  Handles the special cases where a filler fulfills several types of relation.
   */
  private Set<KBPSlot> extractAmbiguousRelations(String entityName, String entityId, String fillerName, Set<String> relations) {
    Set<KBPSlot> result = new HashSet<KBPSlot>();
    if (relations.contains("per:title")) {
      String[] split = fillerName.split(" ((in)|(of)|(with)) ");
      if (split.length > 1) {
        result.addAll(extractUnambiguousRelation(entityName, entityId, split[0], "per:title"));
        if (relations.contains("per:member_of"))
          result.addAll(extractUnambiguousRelation(entityName, entityId, split[1], "per:member_of"));
        if (relations.contains("per:employee_of"))
          result.addAll(extractUnambiguousRelation(entityName, entityId, split[1], "per:employee_of"));
      }
    } else if (relations.contains("org:founded") && relations.contains("org:dissolved")) {
      String[] split = fillerName.split("-");
      if (split.length > 1) {
        result.addAll(extractUnambiguousRelation(entityName, entityId, split[0], "org:founded"));
        result.addAll(extractUnambiguousRelation(entityName, entityId, split[1], "org:dissolved"));
      }
    } else if (relations.contains("org:member_of") && relations.contains("per:employee_of")) {
      String[] names = fillerName.split("\\s*,\\s+|\\s+and\\s+|\\s+or\\s+|\\n|\\t");
      for (String name : names) {
        result.addAll(extractUnambiguousRelation(entityName, entityId, name, "org:member_of"));
        result.addAll(extractUnambiguousRelation(entityName, entityId, name, "per:employee_of"));
      }
    } else if (relations.contains("org:member_of") && relations.contains("org:politicalSLASHreligious_affiliation")) {
      String[] names = fillerName.split("\\s*,\\s+|\\s+and\\s+|\\s+or\\s+|\\n|\\t");
      for (String name : names) 
        result.addAll(extractUnambiguousRelation(entityName, entityId, name, "org:politicalSLASHreligious_affiliation"));
    }	else if (relations.contains("per:date_of_birth") || relations.contains("per:date_of_death")) {
      String type = "birth";
      if (relations.contains("per:date_of_death")) type = "death";
      Pattern pattern = Pattern.compile("[A-Za-z0-9, ]*[0-9]");
      Matcher matcher = pattern.matcher(fillerName);
      if (matcher.find()) {
        result.add(new KBPSlot(entityName, entityId, matcher.group(), "per:date_of_" + type));
        String location = fillerName.substring(matcher.end()).trim();
        result.addAll(extractLocationRelation(entityName, entityId, location, type));
      }
    } else if (relations.contains("per:country_of_birth")) {
      result.addAll(extractLocationRelation(entityName, entityId, fillerName, "birth"));
    } else if (relations.contains("per:country_of_death")) {
      result.addAll(extractLocationRelation(entityName, entityId, fillerName, "death"));
    } else if (relations.contains("per:countries_of_residence")) {
      result.addAll(extractLocationRelation(entityName, entityId, fillerName, "residence"));
    } else if (relations.contains("org:country_of_headquarters")) {
      result.addAll(extractLocationRelation(entityName, entityId, fillerName, "headquarters"));
    }
    return result;
  }
  
  private Set<KBPSlot> extractLocationRelation(String entityName, String entityId, String fillerName, String locationType) {
    Set<KBPSlot> result = new HashSet<KBPSlot>();
    if (!fillerName.matches("[A-Z].*[A-Za-z]") || fillerName.contains(" and ")) return result;
  	
    String[] split = fillerName.split(",\\W+");
    String perORorg = "per";
    if (locationType.equals("headquarters")) perORorg = "org";
    KBPSlot rel;
    
    if (split.length == 1) {
      if (countries.contains(split[0])){
        rel = new KBPSlot(entityName, entityId, split[0], perORorg + ":country_of_" + locationType);
        result.add(rel);
      }
      else if (statesAndProvinces.contains(split[0])){
        rel = new KBPSlot(entityName, entityId, split[0], perORorg + ":stateorprovince_of_" + locationType);
        result.add(rel);
      }
    } 
    
    else if (split.length == 2) {
      rel = new KBPSlot(entityName, entityId, split[0], perORorg + ":city_of_" + locationType);
      result.add(rel);
      
      if (statesAndProvinces.contains(split[1])){
        rel = new KBPSlot(entityName, entityId, split[1], perORorg + ":stateorprovince_of_" + locationType);
        result.add(rel);
      }
      else if (countries.contains(split[1])){
        rel = new KBPSlot(entityName, entityId, split[1], perORorg + ":country_of_" + locationType);
        result.add(rel);
      }
    } 
    
    else if (split.length == 3) {
      rel = new KBPSlot(entityName, entityId, split[0], perORorg + ":city_of_" + locationType);
      result.add(rel);
      
      if (statesAndProvinces.contains(split[1])){
        rel = new KBPSlot(entityName, entityId, split[1], perORorg + ":stateorprovince_of_" + locationType);
        result.add(rel);
      }
      if (countries.contains(split[2])){
        rel = new KBPSlot(entityName, entityId, split[2], perORorg + ":country_of_" + locationType);
        result.add(rel);
      }
    }
    return result;
  }


  //////////////// convenience methods for parsing the xml document ///////////////////
  private static String getText(Node node) {
    return node.getChildNodes().item(0).getNodeValue();
  }

  private static String getTextWithLinks(Node node) {
    NodeList children = node.getChildNodes();
    String result = "";

    for (int i = 0; i < children.getLength(); i++) {
      Node child = children.item(i);
      if (child.getNodeName().equals("link") && child.getChildNodes().getLength() > 0)
        result += "<link>" + getText(child) +"</link>";
      else
        result += child.getNodeValue();
    }   
    return result;

  }

  private static String getTextWOLinks(Node node) {
    NodeList children = node.getChildNodes();
    String result = "";

    for (int i = 0; i < children.getLength(); i++) {
      Node child = children.item(i);
      if (child.getNodeName().equals("link") && child.getChildNodes().getLength() > 0)
        result += getText(child);
      else
        result += child.getNodeValue();
    }	
    return result;
  }

  public static String removeParentheses(String name) {
    String[] split = name.split("\\([^\\(\\)]*\\)");
    String result = "";

    for (String frag : split)
      result += frag;
    return result;
  }

  ///////////////// methods to read from files ////////////////////

  /**
   * Using the mapping files in MAP_DIR, populates both the infoboxToKBP map and the set
   * of possible entity classes
   */
  private void readMapFiles() throws IOException {
    File dir = new File(MAP_DIR);

    for (File file : dir.listFiles()) {
      EntityType entType = null;
      if (file.getName().startsWith("per_")) 
        entType = EntityType.PERSON;
      else if(file.getName().startsWith("org_")) 
        entType = EntityType.ORGANIZATION;
      else throw new RuntimeException("Cannot map file " + file.getName() + " to an entity type!");
      
      BufferedReader rd = new BufferedReader(new FileReader(file));
      while (true) {
        String line = rd.readLine();
        if (line == null) break;
        
        String[] mapping = line.split("\t");
        
        // standardize some of the names in the data set
        mapping[1] = mapping[1].replace("state_or_province", "stateorprovince");
        mapping[1] = mapping[1].replace("/", "SLASH");
        
        // since infoboxToKBP is essentially a multimap, we have to
        // add key-value pairs in a special way
        if (infoboxToKBP.containsKey(mapping[0])) {
          infoboxToKBP.get(mapping[0]).add(mapping[1]);
        } else {
          Set<String> wrapper = new HashSet<String>();
          wrapper.add(mapping[1]);
          infoboxToKBP.put(mapping[0], wrapper);
        } 
        
        String entClass = mapping[0].split(":")[0].toLowerCase();
        entityClasses.add(entClass);
        if (entType == EntityType.PERSON) 
          perEntityClasses.add(entClass);
        else if(entType == EntityType.ORGANIZATION) 
          orgEntityClasses.add(entClass);
      }
      rd.close();
    }
  }
  
  @SuppressWarnings("unused") // previously called at the end of readMapFiles()
  private void checkForOverlaps() {
    for(String per: perEntityClasses) {
      if(orgEntityClasses.contains(per)) {
        System.err.println("Found overlap between PER and ORG for class: " + per);
      }
    }
  }

  /**
   * Using the file NER_TYPES and the files in the MANUAL_LISTS directory, populates the NERTypes map.
   */
  private void readNERTypes() throws IOException {
    BufferedReader rd = new BufferedReader(new FileReader(NER_TYPES));
    while (true) {
      String line = rd.readLine();
      if (line == null) break;
      
      String[] mapping = line.split("\t");
      
      /* standardize some of the names in the data set */
      mapping[0] = mapping[0].replace("/", "SLASH");
      
      if (mapping.length >= 3) {
        Set<String> wrapper = new HashSet<String>();
        String[] split = mapping[2].split("/");
        for (String NERtype : split)
          wrapper.add(NERtype);
        NERTypes.put(mapping[0], wrapper); 
      }
    }
    rd.close();
    
    File dir = new File(MANUAL_LISTS);
    for (File file : dir.listFiles()) {
      rd = new BufferedReader(new FileReader(file));
      String relation = file.getName();
      Set<String> possFillers = new HashSet<String>();
      while (true) {
        String line = rd.readLine();
        if (line == null) break;
        possFillers.add(line.toLowerCase());
      }
      NERTypes.put(relation, possFillers);
      rd.close();
    }
  }


  private void readLocations() throws IOException {
    BufferedReader rd = new BufferedReader(new FileReader(COUNTRIES));
    while (true) {
      String line = rd.readLine();
      if (line == null) break;
      countries.add(line.split("\t")[1]);
    }
    rd.close();

    rd = new BufferedReader(new FileReader(STATES_AND_PROVINCES));
    while (true) {
      String line = rd.readLine();
      if (line == null) break;
      statesAndProvinces.add(line);
    }
    rd.close();
  }

  ///////////////// methods for outputting relations ////////////////////
  
  /**
   * Creates a map from relation types to the associated RelationMention objects. Adding entries to the 
   * map must be handled in a special way, since result is essentially a multimap. 
   */
  private Map<String, Set<KBPSlot>> convertToMap(Set<KBPSlot> relationMentions) {
    
    Map<String, Set<KBPSlot>> result = new HashMap<String, Set<KBPSlot>>();
    for (KBPSlot mention : relationMentions) {
      String key = mention.slotName;
      
      if (result.containsKey(key)) {
        result.get(key).add(mention);
      } else {
        Set<KBPSlot> wrapper = new HashSet<KBPSlot>();
        wrapper.add(mention);
        result.put(key, wrapper);
      } 
    }
    return result;
  }

  
  /**
   * Writes each RelationMention to a file, where each file holds a different type of relation.
   * Should only be called after one or more calls to reader.parseDocument().
   */
  private void writeRelationMentions(String kbDebug) throws IOException {
    Map<String, Set<KBPSlot>> map = convertToMap(relationMentions);
    
    BufferedWriter wr;
    for (String relationName: map.keySet()) {
      wr = new BufferedWriter(new FileWriter(kbDebug + File.separator + relationName));
      for (KBPSlot mention : map.get(relationName))
        wr.write(mention.slotName + "\t" + mention.entityName + "\t" + mention.slotValue + "\n");
      wr.close();
    }
  }
  
  private void normalizeRelationName(KBPSlot rel) {
    if(rel.slotName.endsWith("country_of_residence")) 
      rel.slotName = rel.slotName.substring(0, 4) + "countries_of_residence";
    else if(rel.slotName.endsWith("stateorprovince_of_residence")) 
      rel.slotName = rel.slotName.substring(0, 4) + "stateorprovinces_of_residence";
    else if(rel.slotName.endsWith("city_of_residence")) 
      rel.slotName = rel.slotName.substring(0, 4) + "cities_of_residence";
  }
  
  /**
   * Outputs a map from entity names to RelationMentions concerning that entity. Should only
   * be called after one or more calls to reader.parse()
   */
  private Map<KBPEntity, List<KBPSlot>> getRelationsMap() {
    Map<KBPEntity, List<KBPSlot>> result = new HashMap<KBPEntity, List<KBPSlot>>();
    for (KBPSlot mention : relationMentions) {
      KBPEntity em = new KBPEntity();
      em.name = mention.entityName;
      em.id = mention.entityId;
      em.type = mention.entityType;
      assert(em.id != null);
      
      normalizeRelationName(mention);
      // just a sanity check
      if(mention.slotName.equals("per:city_of_residence")){
        throw new RuntimeException("Invalid relation: " + mention.entityName + "/" + mention.slotName + "/" + mention.slotValue);
      }
     
      if (result.containsKey(em)) {
        result.get(em).add(mention);
      } else {
        List<KBPSlot> wrapper = new ArrayList<KBPSlot>();
        wrapper.add(mention);
        result.put(em, wrapper);
      } 
    }
    return result;
  }
  
  //////////////////// main method /////////////////
  
  public static void main(String [] argv) throws Exception {
    Properties props = StringUtils.argsToProperties(argv);
    KBPDomReader reader = new KBPDomReader(props);
    //Log.setLevel(Level.FINEST);
    
    // parse each document in the KB
    String kbDir = props.getProperty("kbp.inputkb", "/u/nlp/data/TAC-KBP2010/TAC_2009_KBP_Evaluation_Reference_Knowledge_Base/data");
    reader.parse(kbDir);
        
    // save all relations for debug
    String kbDebug = props.getProperty("kbp.debugkb", "/u/nlp/data/TAC-KBP2010/clean_knowledge_base/data/");
    reader.writeRelationMentions(kbDebug);
  }
}