package edu.stanford.nlp.kbp.slotfilling.common;

import java.io.BufferedReader;
import java.io.FileReader;

public class AddCountryLabel {
  public static void main(String[] args) throws Exception {
    BufferedReader is = new BufferedReader(new FileReader(args[0]));
    for(String line; (line = is.readLine()) != null; ) {
      System.out.println(line + "\tCOUNTRY\tLOCATION");
    }
  }
}
