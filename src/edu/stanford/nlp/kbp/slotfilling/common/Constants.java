package edu.stanford.nlp.kbp.slotfilling.common;

public class Constants {
  // whether to use anydoc scoring by default
  public static final boolean DEFAULT_ANYDOC = true;

  public static boolean CASE_INSENSITIVE_ENTITY_MATCH = false;
  
  public static boolean CASE_INSENSITIVE_SLOT_MATCH = false;
  
  public static boolean EXACT_SLOT_MATCH = false;
  
  public static boolean EXACT_ENTITY_MATCH = false;
  
  /** Default model type */
  public static String DEFAULT_MODEL = "lr_inc";
  public static String DEFAULT_ATLEASTONCE_MODEL = "atleastonce_inc";
    
  /** 
   * Ratios of neg/pos when doing subsampling of negatives
   * Decrease for recall, increase for precision
   * This is ratio of negative examples / total positive examples 
   */
  public static double DEFAULT_NEGATIVES_SAMPLING_RATIO = 0.50;
  
  // when tuning slot thresholds fails (due to little/no data), we fall back to this threshold value
  public static final double DEFAULT_SLOT_THRESHOLD = 0.50;
  
  /** If true, we do static subsampling of negatives, before training */
  public static boolean OFFLINE_NEGATIVES = true;
  
  /** Maximum distance between entity and slot candidate, in tokens */
  public static final int MAX_DISTANCE_BETWEEN_ENTITY_AND_SLOT = 20;
    
  /** Should we use old-style sentence caching, from KBP 2010? */
  public static final boolean USE_OLD_CACHING = false;
  
  /** If we are looking at temporal slot values */
  public static final boolean USE_TEMPORAL = false;
  
  /** Should we fetch entity matches using coreference chains? */
  public static final boolean USE_COREF = true;
  
  /** Treat COUNTRY slots as similar to NATIONALITY slots? */
  public static final boolean COUNTRY_EQ_NATIONALITY = false;
  
  //TODO: this should be a constant elsewhere; find that constant
  public static final String NER_BLANK_STRING = "O";
  
  /** Softmax parameter for relation classifier */
  public static final double SOFTMAX_GAMMA = 1.0;
    
  public static final String WEBINDEX_NAME = "WEBSNIPPET";

  /** Extension for the serialized extractor models */
  public static final String SER_EXT = ".ser";
  
  public static String getIndexPath(String path) {
    if(path == null) return WEBINDEX_NAME;
    return path;
  }
  
  /**
   * Converts an index path into a canonical domain name
   * @param indexPath
   */
  public static String indexToDomain(String indexPath, String style) {
    if(style.equals("all")){
      // TODO: this needs to be adjusted if we change indices. Better yet, it should be a config file
      if(indexPath.equals(WEBINDEX_NAME)) return "DWEB";
      if(indexPath.contains("TAC_2010_KBP_Source_Data")) return "DCORPUS";
      if(indexPath.contains("TAC_2009_KBP_Evaluation_Reference_Knowledge_Base")) return "DKB";
      if(indexPath.contains("lr_en_100622")) return "DWIKI";
      throw new RuntimeException("ERROR: Unknown index: " + indexPath);
    } else if(style.equals("two")) {
      if(indexPath.equals(WEBINDEX_NAME)) return "DWEB";
      return "DNONWEB";
    } else if(style.equals("three")) {
      // TODO: this needs to be adjusted if we change indices. Better yet, it should be a config file
      if(indexPath.equals(WEBINDEX_NAME)) return "DWEB";
      if(indexPath.contains("TAC_2010_KBP_Source_Data")) return "DCORPUS";
      if(indexPath.contains("TAC_2009_KBP_Evaluation_Reference_Knowledge_Base")) return "DWIKI";
      if(indexPath.contains("lr_en_100622")) return "DWIKI";
      throw new RuntimeException("ERROR: Unknown index: " + indexPath);
    } else {
      throw new RuntimeException("ERROR: Unknown domain adaptation style: " + style);
    }
  }
}
