package edu.stanford.nlp.kbp.slotfilling.common;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class Nationalities {
  private Map<String, String> countryToNationality;
  private Map<String, String> nationalityToCountry;
  
  public String nationalityToCountry(String s) {
    s = s.toLowerCase();
    if(nationalityToCountry.containsKey(s)) return nationalityToCountry.get(s);
    return null;
  }
  
  public String countryToNationality(String s) {
    s = s.toLowerCase();
    if(countryToNationality.containsKey(s)) return countryToNationality.get(s);
    return null;
  }
  
  private void loadNationalities(String fn) throws IOException {
    nationalityToCountry = new HashMap<String, String>();
    countryToNationality = new HashMap<String, String>();
    
    BufferedReader is = new BufferedReader(new FileReader(fn));
    for(String line; (line = is.readLine()) != null; ) {
      String [] bits = line.split("\\s+");
      nationalityToCountry.put(bits[1].toLowerCase().replaceAll("_", " "), bits[0].replaceAll("_", " "));
      countryToNationality.put(bits[0].toLowerCase().replaceAll("_", " "), bits[1].replaceAll("_", " "));
    }
    is.close();

    Log.severe("Loaded " + nationalityToCountry.size() + " nationalities.");
  }
  
  public Nationalities(String fn) throws IOException {
    loadNationalities(fn);
  }
}
