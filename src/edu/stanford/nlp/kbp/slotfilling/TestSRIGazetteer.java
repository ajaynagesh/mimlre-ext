package edu.stanford.nlp.kbp.slotfilling;

import java.util.List;

import com.sri.faust.gazetteer.*;
import com.sri.faust.gazetteer.maxmind.*;
import com.sri.faust.gazetteer.geocommons.*;
import com.sri.faust.gazetteer.maxmind.region.*;

@SuppressWarnings("unused")
public class TestSRIGazetteer {
  public static void main(String[] args) throws Exception {
    //first step is to create a Gazetteer object
    Gazetteer gazetteerObject = new MaxmindGazetteer();
    
  //Now that we have a gazetter object lets find all the towns named Helena
    List<GazetteerCity> cities = gazetteerObject.getCitiesWithName("Helena");
    
    //there are more then one so lets print out how many we have found
    System.out.println("Number of Helena's found: " + cities.size());
    GazetteerCity currentCity = null;
    //Lets find the listing for Helena, Montana
    //We know that Montana, USA has a region name of Montana
    for(int index = 0; index < cities.size(); index++)
    {
        currentCity = cities.get(index);
        
        if(currentCity.getRegionName().contains( "Montana"))
        {break;}
    }
    
    //Now lets print out its population, latitude and longitude
    System.out.println("Helena, MT population: " + currentCity.getPopulation());
    System.out.println("Helena, MT Latitude: " + currentCity.getLatitudeDegrees());
    System.out.println("Helena, MT Longitude: " + currentCity.getLongitudeDegrees());
    
    //lets get the region Helena MT belongs to and print out the Region name
    GazetteerRegion helenaRegion = currentCity.getRegion();
    System.out.println("Helena's Region Name: " + helenaRegion.getName());
    
    //Now lets get the country info for The United States by URI
    GazetteerCountry country = gazetteerObject.getCountryFromURI("http://dbpedia.org/resource/United_States");
    
    //Lets print out the Countries name and latitude and longitude
    System.out.println(country.getName() + ": ");
    System.out.println(country.getLatitudeDegrees());
    System.out.println(country.getLongitudeDegrees());
    
    List<GazetteerCountry> countries = gazetteerObject.getCountryFromRegion("Texas");
    System.out.println("Countries found for the state Texas:");
    for(GazetteerCountry c: countries) {
      System.out.println("\t" + c.getName());
    }
    
    List<GazetteerRegion> regions = gazetteerObject.getRegionFromRegion("Texas");
    System.out.println("Found " + regions.size() + " regions with the name Texas.");
  }
}
