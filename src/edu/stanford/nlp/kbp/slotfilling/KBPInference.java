package edu.stanford.nlp.kbp.slotfilling;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.sri.faust.gazetteer.Gazetteer;
import com.sri.faust.gazetteer.GazetteerCity;
import com.sri.faust.gazetteer.GazetteerRegion;

import edu.stanford.nlp.kbp.slotfilling.common.KBPSlot;
import edu.stanford.nlp.kbp.slotfilling.common.ListOutput;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.SlotType;
import edu.stanford.nlp.kbp.slotfilling.common.SlotsToNamedEntities;

public class KBPInference {
  private final SlotsToNamedEntities slotsToNamesEntities;
  
  private final ListOutput listOutput;
  
  private final Gazetteer gazetteer;
  
  private final Map<String, Set<String>> stateAbbreviations;
  private final Map<String, String> stateAbbrevToFull;
  
  private final boolean doDomainSpecificInference;
  
  public KBPInference(SlotsToNamedEntities s, 
      ListOutput listOutput, 
      Gazetteer gaz, 
      Map<String, Set<String>> abbrevs,
      Map<String, String> abbrevToFull,
      boolean doDomainSpec) {
    this.slotsToNamesEntities = s;
    this.listOutput = listOutput;
    this.gazetteer = gaz;
    this.stateAbbreviations = abbrevs;
    this.stateAbbrevToFull = abbrevToFull;
    this.doDomainSpecificInference = doDomainSpec;
  }
  
  private static final Set<String> NEEDS_SPECIAL_INFERENCE = new HashSet<String>(Arrays.asList(
    "per:country_of_birth", "per:stateorprovince_of_birth", "per:city_of_birth",
    "per:country_of_death", "per:stateorprovince_of_death", "per:city_of_death",
    "per:date_of_birth", "per:date_of_death", 
    "org:subsidiaries", "org:parents",
    "org:founded", "org:dissolved",
    "org:country_of_headquarters", "org:stateorprovince_of_headquarters", "org:city_of_headquarters"
  ));
  
  private static final String NIL = "NIL";
  
  public Collection<KBPSlot> inference(Collection<KBPSlot> slots) {
    Collection<KBPSlot> results = new ArrayList<KBPSlot>();
    
    //
    // build the buckets, one for each slot name
    // also, keep track of the best value for each slot name
    //
    Map<String, List<KBPSlot>> slotBuckets = new HashMap<String, List<KBPSlot>>();
    Map<String, KBPSlot> bestPerSlot = new HashMap<String, KBPSlot>();
    for(KBPSlot slot: slots) {
      List<KBPSlot> sameNameSlots = slotBuckets.get(slot.slotName);
      if(sameNameSlots == null){
        sameNameSlots = new ArrayList<KBPSlot>();
        slotBuckets.put(slot.slotName, sameNameSlots);
      }
      sameNameSlots.add(slot);
      
      if (! bestPerSlot.containsKey(slot.slotName) || 
          slot.getScore() > bestPerSlot.get(slot.slotName).getScore()) {
        bestPerSlot.put(slot.slotName, slot);
      } else if(slot.getScore() == bestPerSlot.get(slot.slotName).getScore() &&
          slot.slotValue.compareTo(bestPerSlot.get(slot.slotName).slotValue) < 0) {
        // arbitrarily choose one slot to solve ties
        bestPerSlot.put(slot.slotName, slot);
      }
    }
    
    //
    // handle the slots that do not require special inference
    //
    for(String slotName: slotBuckets.keySet()){
      if(! doDomainSpecificInference || ! NEEDS_SPECIAL_INFERENCE.contains(slotName)){
        // for SINGLE slots keep the best value
        if(slotsToNamesEntities.getSlotInfo(slotName).slotType() == SlotType.SINGLE) {
          KBPSlot rel = bestPerSlot.get(slotName);
          assert(rel != null);
          results.add(rel);
        }
        
        // for LIST slots keep all values (if listOutput == ALL) or just the best otherwise
        else if(listOutput == ListOutput.ALL) {
          List<KBPSlot> rels = slotBuckets.get(slotName);
          assert(rels != null);
          results.addAll(rels);
        }
        
        else if(listOutput == ListOutput.BEST) {
          KBPSlot rel = bestPerSlot.get(slotName);
          assert(rel != null);
          results.add(rel);
        }
        
        else if(listOutput == ListOutput.TOP) {
          // keep just the top slots over a certain threshold
          throw new RuntimeException("ERROR: list output style TOP not supported!");
        }
      }
    }
    
    if(doDomainSpecificInference){
      //
      // run the special, domain-specific inference
      //
      CityStateCountryInference csci = new CityStateCountryInference(gazetteer, stateAbbreviations, stateAbbrevToFull);
      results.addAll(csci.choose(slotBuckets,
          "per:city_of_birth",
          "per:stateorprovince_of_birth",
          "per:country_of_birth"));
      results.addAll(csci.choose(slotBuckets,
          "per:city_of_death",
          "per:stateorprovince_of_death",
          "per:country_of_death"));
      results.addAll(csci.choose(slotBuckets,
          "org:city_of_headquarters",
          "org:stateorprovince_of_headquarters",
          "org:country_of_headquarters"));

      DateIntervalInference dii = new DateIntervalInference();
      results.addAll(dii.choose(
          slotBuckets.get("per:date_of_birth"),
          slotBuckets.get("per:date_of_death")));
      results.addAll(dii.choose(
          slotBuckets.get("org:founded"),
          slotBuckets.get("org:dissolved")));

      DistinctSetInference dsi = new DistinctSetInference();
      results.addAll(dsi.choose(
          slotBuckets.get("org:subsidiaries"),
          slotBuckets.get("org:parents")));
    }
    
    return results;
  }
  
  static class Candidate {
    List<KBPSlot> solution;
    double score;
    
    public Candidate() {
      solution = new ArrayList<KBPSlot>();
      score = 0;
    }
    
    public Collection<KBPSlot> prepareSolution() {
      Collection<KBPSlot> result = new ArrayList<KBPSlot>();
      for(KBPSlot slot: solution) {
        if(! slot.slotValue.equals(KBPInference.NIL)) result.add(slot);
      }
      return result;
    }
    
    @Override
    public String toString() {
      StringBuffer os = new StringBuffer();
      os.append("[");
      boolean first = true;
      for(KBPSlot slot: solution) {
        if(! first) os.append(", ");
        os.append("(" + slot + ")");
        first = false;
      }
      os.append("] (" + score + ")");
      return os.toString();
    }
    
    public static Candidate makeCandidate(Candidate current, KBPSlot cell) {
      Candidate newCand = new Candidate();
      newCand.solution.addAll(current.solution);
      newCand.solution.add(cell);
      newCand.score = current.score + cell.getScore();
      return newCand;
    }
  };
  
  private static KBPSlot makeEmptySlot() {
    KBPSlot slot = new KBPSlot(KBPInference.NIL, KBPInference.NIL, KBPInference.NIL, KBPInference.NIL);
    slot.setScore(0);
    return slot;
  }
  
  private static void makeCandidates(Candidate current, 
      int columnPosition, 
      List<List<KBPSlot>> columns, 
      List<Candidate> candidates) {
    // reached the end of the column traversal
    if(columnPosition >= columns.size()) {
      assert(current.solution.size() == columns.size());
      candidates.add(current);
      return;
    }
    
    // traverse the current column and create a new candidate for each cell
    for(int i = 0; i < columns.get(columnPosition).size(); i ++) {
      Candidate newCand = Candidate.makeCandidate(current, columns.get(columnPosition).get(i));
      makeCandidates(newCand, columnPosition + 1, columns, candidates);
    }
  }
  
  
  public static List<Candidate> prepareCandidates(List<KBPSlot> ... args) {
    //
    // prepare the columns:
    // - make empty ones for NILs
    // - add a NIL slot at the beginning of all of them, so we can have solutions with empty cells
    //
    List<List<KBPSlot>> columns = new ArrayList<List<KBPSlot>>();
    for(List<KBPSlot> arg: args) {
      if(arg == null) arg = new ArrayList<KBPSlot>();
      arg.add(0, makeEmptySlot());
      columns.add(arg);
    }
    
    // exact inference: recursively create all possible combos 
    List<Candidate> candidates = new ArrayList<Candidate>();
    Candidate seed = new Candidate();
    makeCandidates(seed, 0, columns, candidates);
    
    // sort in descending order of scores
    Collections.sort(candidates, new Comparator<Candidate>() {
      @Override
      public int compare(Candidate o1, Candidate o2) {
        if(o1.score > o2.score) return -1;
        if(o1.score == o2.score) return 0;
        return 1;
      }
    });
    
    return candidates;
  }
  
  private static class CityStateCountryInference {
    private final Gazetteer gazetteer;
    private final Map<String, Set<String>> stateAbbreviations;
    private final Map<String, String> stateAbbrevToFull;
    
    public CityStateCountryInference(Gazetteer gaz, Map<String, Set<String>> abbrevs, Map<String, String> stateAbbrevToFull) {
      this.gazetteer = gaz;
      this.stateAbbreviations = abbrevs;
      this.stateAbbrevToFull = stateAbbrevToFull;
    }
    
    @SuppressWarnings("unchecked")
    public Collection<KBPSlot> choose(Map<String, List<KBPSlot>> slotBuckets, String citySlot,
        String stateSlot, String countrySlot) {
      List<KBPSlot> cities = slotBuckets.get(citySlot);
      List<KBPSlot> states = slotBuckets.get(stateSlot);
      List<KBPSlot> countries = slotBuckets.get(countrySlot);
      
      // exact inference: generate all possible solutions, sorted by overall score
      List<Candidate> candidates = KBPInference.prepareCandidates(cities, states, countries);
      
      // pick the first one that is consistent with the location constraints
      for(int i = 0; i < candidates.size(); i ++){
        Candidate cand = candidates.get(i);
        
        // infer region from city/country, for example (no longer used -- see comment on the method)
        // fillInHoles(cand, citySlot, stateSlot, countrySlot);
        
        if (consistent(cand)) {
          Log.severe("LOCATION INFERENCE: accepted consistent solution: " + cand);
          if(i > 0) Log.severe("LOCATION INFERENCE: chose solution: " + cand + " versus initial solution: " + candidates.get(0));
          return cand.prepareSolution();
        } else {
          Log.severe("LOCATION INFERENCE: dropped inconsistent solution: " + cand);
        }
      }
      
      return new ArrayList<KBPSlot>();
    }
    
    // this inference turns out to hurt, probably because the underlying RE system is so noisy
    @SuppressWarnings("unused")
    private void fillInHoles(Candidate cand, String citySlotName, String stateSlotName, String countrySlotName) {
      KBPSlot city = cand.solution.get(0);
      KBPSlot state = cand.solution.get(1);
      KBPSlot country = cand.solution.get(2);
      
      boolean haveCity = city != null && !city.slotValue.equals(NIL);
      boolean haveState = state != null && !state.slotValue.equals(NIL);
      boolean haveCountry = country != null && !country.slotValue.equals(NIL);

      // fill in state from city and country
      if (!haveState && haveCity && haveCountry) {
        List<GazetteerCity> cities = findCities(city.slotValue);

        if (cities.size() > 0) {
          Collections.sort(cities, new CitiesByPopulation());
          
          for (GazetteerCity c : cities) {
            if (compatibleCityCountry(c, country.slotValue)) {
              String regionName = fullRegionName(c.getRegionName());
              KBPSlot regionSlot = new KBPSlot(city.entityName, city.entityId, regionName, stateSlotName);
              cand.solution.set(1, regionSlot);
              Log.severe("LOCATION INFERENCE: detected region " + regionName + " from city " + city.slotValue + " / country " + country.slotValue);
              break;
            }
          }
        }
      }
      
      // fill in country
      if (!haveCountry) {
        if (haveCity && haveState) {
          // fill in country from city and state
          List<GazetteerCity> cities = findCities(city.slotValue);

          if (cities.size() > 0) {
            Collections.sort(cities, new CitiesByPopulation());

            for (GazetteerCity c : cities) {
              if (compatibleCityState(c, state.slotValue)) {
                String countryName = c.getCountry().getName();
                KBPSlot newCountrySlot = new KBPSlot(state.entityName, state.entityId, countryName, countrySlotName);
                cand.solution.set(2, newCountrySlot);
                Log.severe("LOCATION INFERENCE: detected country " + countryName + " from city " + city.slotValue + " / state " + state.slotValue);
                break;
              }
            }
          }
        } else if (haveState) {
          // fill in country just from state
          List<GazetteerRegion> regions = gazetteer.getRegionFromRegion(fullRegionName(state.slotValue));
          
          if (regions.size() > 0) {
            // TODO should sort regions by frequency or something, but don't have that data currently
            // for now, we arbitrarily pick the first one after sorting by name
            Collections.sort(regions, new RegionsByName());
            GazetteerRegion region = regions.get(0);
            String countryName = region.getCountry().getName();
            KBPSlot newCountrySlot = new KBPSlot(state.entityName, state.entityId, countryName, countrySlotName);
            cand.solution.set(2, newCountrySlot);
            Log.severe("LOCATION INFERENCE: detected country " + countryName + " from state " + state.slotValue);
          }
        } else if (haveCity) {
          // fill in country (and optionally region) just from city
          List<GazetteerCity> cities = findCities(city.slotValue);

          if (cities.size() > 0) {
            Collections.sort(cities, new CitiesByPopulation());

            for (GazetteerCity c : cities) {
              String countryName = c.getCountry().getName();
              KBPSlot newCountrySlot = new KBPSlot(city.entityName, city.entityId, countryName, countrySlotName);
              cand.solution.set(2, newCountrySlot);
              Log.severe("LOCATION INFERENCE: detected country " + countryName + " from city " + city.slotValue);
              
              if (!haveState) {
                String regionName = fullRegionName(c.getRegionName());
                KBPSlot newRegionSlot = new KBPSlot(city.entityName, city.entityId, regionName, stateSlotName);
                cand.solution.set(1, newRegionSlot);
                Log.severe("LOCATION INFERENCE: detected region " + regionName + " from city " + city.slotValue);
              }
              
              break;
            }
          }
        }
      }
    }

    private String fullRegionName(String s) {
      String sl = s.toLowerCase();
      if(stateAbbrevToFull.containsKey(sl)){
        return stateAbbrevToFull.get(sl);
      }
      return s;
    }
    
    private List<GazetteerCity> findCities(String name) {
      List<GazetteerCity> cities = gazetteer.getCitiesWithName(name);
      if (cities != null && cities.size() > 0) {
        return cities;
      }
      
      // try again after remove " city" suffix
      if (name.toLowerCase().endsWith("city")) {
        name = name.substring(0, name.length() - 4).trim();
        return gazetteer.getCitiesWithName(name);
      }
      
      return new ArrayList<GazetteerCity>();
    }
    
    public boolean consistent(Candidate cand) {
      KBPSlot city = cand.solution.get(0);
      KBPSlot state = cand.solution.get(1);
      KBPSlot country = cand.solution.get(2);
      
      if(city != null && ! city.slotValue.equals(NIL)) {
        List<GazetteerCity> cities = findCities(city.slotValue);
        if(cities.size() > 0){
          // known city; check its state/country
          for(GazetteerCity c: cities){
            if(compatibleCityState(c, state.slotValue) && compatibleCityCountry(c, country.slotValue)){
              return true;
            }
          }
          return false;
        } else {
          // unknown city; do not accept these
          return false;
        }
      }
      
      if(state != null && ! state.slotValue.equals(NIL)){
        List<GazetteerRegion> regions = gazetteer.getRegionFromRegion(fullRegionName(state.slotValue));
        if(regions.size() > 0){
          // known state; check its country
          for(GazetteerRegion r: regions) {
            if(compatibleStateCountry(r, country.slotValue)){
              return true;
            }
          }
          return false;
        } else {
          // unknown state; do not accept these
          return false;
        }
      }
      
      if(country != null && ! country.slotValue.equals(NIL)){
        if(gazetteer.getCountryFromName(country.slotValue) != null || otherCountry(country.slotValue)){
          return true;
        } else {
          // accept only known countries
          return false;
        }
      }
      
      return true;
    }
    
    private Set<String> normalizeRegion(String r) {
      Set<String> normed = new HashSet<String>();
      normed.add(r);
      Set<String> abbrevs = stateAbbreviations.get(r);
      if(abbrevs != null){
        for(String abbrev: abbrevs){
          normed.add(abbrev);
        }
      }
      return normed;
    }
    
    private static final Pattern OTHER_COUNTRY = Pattern.compile("(U\\.?\\s*S\\.?\\s*A\\.?)|(United\\s*States\\s*of\\s*America)|(UK)|(Great\\s*Britain)|(Wales)|(Scotland)", Pattern.CASE_INSENSITIVE);
    
    private static boolean otherCountry(String s) {
      if(OTHER_COUNTRY.matcher(s).matches()) return true;
      return false;
    }
        
    private Set<String> normalizeCountry(String c) {
      Set<String> normed = new HashSet<String>();
      normed.add(c.toLowerCase());
      if(c.equalsIgnoreCase("United States")){
        normed.add("US".toLowerCase());
        normed.add("U.S.".toLowerCase());
        normed.add("USA".toLowerCase());
        normed.add("U.S.A.".toLowerCase());
        normed.add("United States of America".toLowerCase());
      } else if(c.equalsIgnoreCase("United Kingdom")) {
        normed.add("UK".toLowerCase());
        normed.add("Great Britain".toLowerCase());
        normed.add("Wales".toLowerCase());
        normed.add("Scotland".toLowerCase());
      }
      return normed;
    }
    
    private static final Pattern UNKNOWN_REGION_ENDING = Pattern.compile("\\-[0-9]+$");
    
    private boolean compatibleCityState(GazetteerCity city, String regionValue) {
      if(regionValue.equals(NIL)) return true;
      String region = city.getRegionName().toLowerCase();      
      regionValue = regionValue.toLowerCase();

      if(region == null || UNKNOWN_REGION_ENDING.matcher(region).find()){
        // no info on this region; e.g., we do not have info on Canada's provinces - these are marked as Canada-01, etc.
        return true;
      }
      
      Set<String> normRegions = normalizeRegion(region);
      if(normRegions.contains(regionValue)){
        return true;
      }
      return false;
    }
    
    private boolean compatibleCityCountry(GazetteerCity city, String countryValue) {
      if(countryValue.equals(NIL)) return true;
      Set<String> normCountries = normalizeCountry(city.getCountry().getName());
      if(normCountries.contains(countryValue.toLowerCase())){
        return true;
      }
      return false;
    }
    
    private boolean compatibleStateCountry(GazetteerRegion region, String countryValue) {
      if(countryValue.equals(NIL)) return true;
      Set<String> normCountries = normalizeCountry(region.getCountry().getName());
      if(normCountries.contains(countryValue.toLowerCase())){
        return true;
      }
      return false;
    }
  }
  
  private static class DateIntervalInference {
    @SuppressWarnings("unchecked")
    Collection<KBPSlot> choose(
        List<KBPSlot> starts, 
        List<KBPSlot> ends) { 
      
      // exact inference: generate all possible solutions, sorted by overall score
      List<Candidate> candidates = KBPInference.prepareCandidates(starts, ends);
      
      // pick the first one that is consistent, i.e., start year < end year
      for(int i = 0; i < candidates.size(); i ++){
        Candidate cand = candidates.get(i);
        if(consistent(cand)) {
          Log.severe("DATE INTERVAL INFERENCE: accepted consistent solution: " + cand);
          if(i > 0) Log.severe("DATE INTERVAL INFERENCE: chose solution: " + cand + " versus initial solution: " + candidates.get(0));
          return cand.prepareSolution();
        } else {
          Log.severe("DATE INTERVAL INFERENCE: dropped inconsistent solution: " + cand);
        }
      }
      
      return new ArrayList<KBPSlot>();
    }
    
    public boolean consistent(Candidate cand) {
      assert(cand.solution.size() == 2);
      KBPSlot start = cand.solution.get(0);
      KBPSlot end = cand.solution.get(1);
      
      if(start.slotValue.equals(NIL) || end.slotValue.equals(NIL)){
        return true;
      }
      
      int startYear = findYear(start.slotValue);
      int endYear = findYear(end.slotValue);
      if(startYear == -1 || endYear == -1) {
        return true;
      }
      
      if(startYear >= endYear) {
        return false;
      }
      
      return true;
    }
    
    static final Pattern YEAR = Pattern.compile("\\d\\d\\d\\d");
    
    private static int findYear(String s) {
      Matcher m = YEAR.matcher(s);
      if(m.find()) {
        String year = m.group();
        return Integer.valueOf(year);
      }
      return -1;
    }
  }
  
  private static class DistinctSetInference {
    private static KBPSlot findSimilar(KBPSlot e, Collection<KBPSlot> s) {
      for(KBPSlot se: s) {
        if(se.sameSlot(e)) {
          return se;
        }
      }
      return null;
    }
    
    Collection<KBPSlot> choose(
        List<KBPSlot> s1, 
        List<KBPSlot> s2) { 
      Collection<KBPSlot> result = new ArrayList<KBPSlot>();
      
      if(s1 == null) s1 = new ArrayList<KBPSlot>();
      if(s2 == null) s2 = new ArrayList<KBPSlot>();
      
      for(KBPSlot e1: s1) {
        KBPSlot e2 = findSimilar(e1, s2);
        if(e2 == null) result.add(e1);
        else {
          // keep the best of the two
          if(e1.getScore() >= e2.getScore()){
            Log.severe("DISTINCT SET INFERENCE: accepted " + e1 + " over " + e2);
            result.add(e1);
          } else {
            Log.severe("DISTINCT SET INFERENCE: accepted " + e2 + " over " + e1);
            result.add(e2);
          }
        }
      }
      
      for(KBPSlot e2: s2) {
        if(findSimilar(e2, s1) == null) {
          result.add(e2);
        }
      }
      
      return result;
    }
  }

  /**
   * Sorts city from largest population to smallest population. For cities with
   * unknown populations, we sort alphabetically by city name.
   */
  public static class CitiesByPopulation implements Comparator<GazetteerCity> {

    public int compare(GazetteerCity o1, GazetteerCity o2) {
      int comparison = -o1.getPopulation().compareTo(o2.getPopulation());
      
      if (comparison == 0) {
        comparison = o1.getName().compareTo(o2.getName());
      }
      
      return comparison;
    }
    
  }
  
  /**
   * Sorts regions by region name, then by country.
   */
  public static class RegionsByName implements Comparator<GazetteerRegion> {

    public int compare(GazetteerRegion o1, GazetteerRegion o2) {
      int comparison = o1.getName().compareTo(o2.getName());
      
      if (comparison == 0) {
        comparison = o1.getCountry().getName().compareTo(o2.getCountry().getName());
      }
      
      return comparison;
    }
    
  }
  
}
