package edu.stanford.nlp.kbp.slotfilling.webqueries;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.stanford.nlp.util.Triple;

/**
 * Breaks a snippet file into multiple smaller files, so we can parallelize WebSnippetProcessor (used for pre-processing of snippets)
 * @author Mihai
 */
public class WebSnippetsSplitter {
  public static void main(String[] args) throws Exception {
    if(args.length != 2){
      System.err.println("Usage: WebSnippetsSplitter <snippets file> <number of shards>");
      System.exit(1);
    }
    String snippetFile = args[0];
    int numberOfShards = Integer.parseInt(args[1]);
    
    BufferedReader is = new BufferedReader(new FileReader(snippetFile));
    // stores unique entity names
    Set<String> ents = new HashSet<String>();
    // stores the line where a given entity was seen for the first time; in file order
    // also stores how many useful sentences we've seen up to here
    List<Triple<String, Integer, Integer>> entitiesToLines = new ArrayList<Triple<String,Integer, Integer>>();
    String line;
    int lineCount = 0;
    int contentLines = 0;
    while((line = is.readLine()) != null){
      lineCount ++;
      line = line.trim();
      if(line.length() == 0){
        continue;
      }

      String [] bits = line.split("\t");
      assert(bits.length > 2);
      String ent = bits[1];
      // System.err.println("Entity: " + ent);
      if(! ents.contains(ent)) entitiesToLines.add(new Triple<String, Integer, Integer>(ent, lineCount, contentLines));
      ents.add(ent);
      while((line = is.readLine()) != null){
        lineCount ++;
        contentLines ++;
        line = line.trim();
        if(line.length() == 0) break;
      }
      contentLines --; // the first line for each block is meta info not an actual snippet
      
    }
    is.close();
    System.err.println("Found " + ents.size() + " unique entities and " + contentLines + " useful snippets.");
    int snippetsPerShard = contentLines / numberOfShards;
    System.err.println("Will use " + snippetsPerShard + " snippets per shard.");
    
    List<Integer> lineBreaks = new ArrayList<Integer>();
    int snippetsSaved = 0;
    for(Triple<String, Integer, Integer> point: entitiesToLines){
      if(point.third >= snippetsSaved + snippetsPerShard){
        lineBreaks.add(point.second - 1);
        snippetsSaved = point.third;
      }
    }
    System.err.println("Will split original file at these lines: " + lineBreaks);
    
    int fileIndex = 1;
    is = new BufferedReader(new FileReader(snippetFile));
    PrintStream os = new PrintStream(new FileOutputStream(snippetFile + "." + fileIndex));
    lineCount = 0;
    while((line = is.readLine()) != null){
      lineCount ++;
      os.println(line);
      if(lineBreaks.size() > 0 && lineCount == lineBreaks.get(0)){
        os.close();
        fileIndex ++;
        lineBreaks.remove(0);
        os = new PrintStream(new FileOutputStream(snippetFile + "." + fileIndex));
      }
    }
    is.close();
    os.close();
    
  }
}
