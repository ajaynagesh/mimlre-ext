package edu.stanford.nlp.kbp.slotfilling;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;

import com.sri.faust.gazetteer.Gazetteer;
import com.sri.faust.gazetteer.GazetteerCity;
import com.sri.faust.gazetteer.GazetteerCountry;
import com.sri.faust.gazetteer.GazetteerRegion;
import com.sri.faust.gazetteer.maxmind.MaxmindGazetteer;

public class SRIGazetteerQuery {
  public static void main(String[] args) throws Exception {
    //first step is to create a Gazetteer object
    Gazetteer gazetteerObject = new MaxmindGazetteer();
    BufferedReader is = new BufferedReader(new InputStreamReader(System.in));
    System.out.print("> ");
    for(String line; (line = is.readLine()) != null; ) {
      List<GazetteerCity> cities = gazetteerObject.getCitiesWithName(line.trim());
      System.out.println("Found " + cities.size() + " matching cities:");
      for(GazetteerCity city: cities) {
        System.out.println("\t" + city.getName());
        GazetteerRegion reg = city.getRegion();
        System.out.println("\t\tRegion: " + (reg != null ? reg.getName() : "NIL"));
        GazetteerCountry country = city.getCountry();
        System.out.println("\t\tCountry: " + (country != null ? country.getName() : "NIL"));
      }
      System.out.print("> ");
    }
  }
}
