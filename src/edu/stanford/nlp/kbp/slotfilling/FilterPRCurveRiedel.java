package edu.stanford.nlp.kbp.slotfilling;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;

public class FilterPRCurveRiedel {
  public static void main(String[] args) throws Exception {
    PrintWriter os = new PrintWriter(args[0] + ".filter");
    BufferedReader is = new BufferedReader(new FileReader(args[0]));
    String prev = null, lastRead = null;
    double prevF1 = 0;
    for(String line; (line = is.readLine()) != null; ) {
      String [] bits = line.split("\\s+");
      double p = Double.valueOf(bits[1]);
      double r = Double.valueOf(bits[2]);
      double f1 = (p != 0 && r != 0 ? 2*p*r/(p+r) : 0);
      if(prev == null || Math.abs(f1 - prevF1) > 0.004) {
        os.println(line);
        prev = line;
        prevF1 = f1;
      }
      lastRead = line;
    }
    if(lastRead != null && ! lastRead.equals(prev)) {
      os.println(lastRead);
    }
    os.close();
    is.close();
  }
}
