package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationSerializer;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Timing;

/**
 * Compares the Generic and the Custom AnnotationSerializers
 */
public class SerializerComparator {
  static final String GENERIC_OUTPUT = "/tmp/generic_kbp.ser";
  static final String CUSTOM_OUTPUT = "/tmp/custom_kbp.ser";
  static final String CUSTOM_OUTPUT_RESAVED = "/tmp/custom_kbp_resaved.ser";
  
  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    String fileToSerialize = props.getProperty("serialize");
    assert(fileToSerialize != null);
    
    AnnotationSerializer generic = new GenericAnnotationSerializer();
    AnnotationSerializer custom = new KBPAnnotationSerializer(true, true);
    
    //
    // Generate serialized files only if they don't exist already
    //
    File gf = new File(GENERIC_OUTPUT);
    File cf = new File(CUSTOM_OUTPUT);
    if(! gf.exists() || ! cf.exists()){
      String text = readText(fileToSerialize);
      StanfordCoreNLP pipe = new StanfordCoreNLP(props);
      Annotation doc = pipe.process(text); 
      OutputStream gs = new FileOutputStream(gf);
      OutputStream cs = new FileOutputStream(cf);
      generic.save(doc, gs);
      custom.save(doc, cs);
      gs.close();
      cs.close();
      System.out.println("Generic serialization saved in file: " + gf);
      System.out.println("Custom serialization saved in file: " + cf);
    } else {
      System.out.println("Serialized files already exist! Will not regenerate them.");
    }
    
    // Read using the generic serializer
    Timing genericTime = new Timing();
    InputStream gis = new FileInputStream(gf); 
    Annotation genericDoc = generic.load(gis);
    System.out.println("Loading using the generic serializer took: " + genericTime.toSecondsString());
    gis.close();
    
    // Read using the custom serializer
    Timing customTime = new Timing();
    InputStream cis = new FileInputStream(cf); 
    Annotation customDoc = custom.load(cis);
    System.out.println("Loading using the custom serializer took: " + customTime.toSecondsString());
    cis.close();
    
    // make sure we read the same number of sents
    assert(genericDoc.get(SentencesAnnotation.class).size() == customDoc.get(SentencesAnnotation.class).size());
    // make sure we read the same number of tokens
    assert(countTokens(genericDoc) == countTokens(customDoc));
    
    // re-save the custom annotation to make sure we get the same output
    OutputStream cs = new FileOutputStream(CUSTOM_OUTPUT_RESAVED);
    custom.save(customDoc, cs);
    cs.close();
    System.out.println("Custom doc was re-saved to file: " + CUSTOM_OUTPUT_RESAVED + ". This file must be identical to " + CUSTOM_OUTPUT);
  }
  
  private static int countTokens(Annotation doc) {
    int count = 0;
    for(CoreMap sent: doc.get(SentencesAnnotation.class)){
      count += sent.get(TokensAnnotation.class).size();
    }
    return count;
  }
  
  private static String readText(String fn) throws Exception {
    StringBuffer os = new StringBuffer();
    BufferedReader is = new BufferedReader(new FileReader(fn));
    String line;
    while((line = is.readLine()) != null){
      os.append(line);
      os.append("\n");
    }
    is.close();
    return os.toString();
  }
}
