package edu.stanford.nlp.kbp.slotfilling.multir;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import edu.stanford.nlp.kbp.slotfilling.multir.DocumentProtos.Relation;

/**
 * Builds cross validation folds from Hoffmann's training corpus
 * @author Mihai
 *
 */
public class BuildFolds {
  public static void main(String[] args) throws Exception {
    String trainFile = args[0];
    int numberOfFolds = Integer.valueOf(args[1]);
    
    InputStream is = new GZIPInputStream(
        new BufferedInputStream
        (new FileInputStream(trainFile)));
    List<Relation> relations = new ArrayList<Relation>();
    Relation r = null;
    while ((r = Relation.parseDelimitedFrom(is)) != null) {
      relations.add(r);
    }
    is.close();
    relations = randomize(relations, 1);
    
    for(int fold = 0; fold < numberOfFolds; fold ++) {
      saveFold(relations, fold, numberOfFolds);
    }
  }
  
  private static int foldStart(int fold, int numberOfFolds, int size) {
    int foldSize = size / numberOfFolds;
    assert(foldSize > 0);
    int start = fold * foldSize;
    assert(start < size);
    return start;
  }
  
  private static int foldEnd(int fold, int numberOfFolds, int size) {
    // padding if this is the last fold
    if(fold == numberOfFolds - 1) 
      return size;
    
    int foldSize = size / numberOfFolds;
    assert(foldSize > 0);
    int end = (fold + 1) * foldSize;
    assert(end <= size);
    return end;
  }
  
  private static void saveFold(List<Relation> relations, int fold, int numberOfFolds) throws IOException {
    int start = foldStart(fold, numberOfFolds, relations.size());
    int end = foldEnd(fold, numberOfFolds, relations.size());
    File dir = new File("fold" + fold);
    dir.mkdir();
    
    OutputStream osTrain = new GZIPOutputStream(new FileOutputStream(dir + File.separator + "train.pb.gz"));
    OutputStream osTest = new GZIPOutputStream(new FileOutputStream(dir + File.separator + "test.pb.gz"));
    for(int i = start; i < end; i ++){
      relations.get(i).writeDelimitedTo(osTest);
    }
    for(int i = 0; i < start; i ++){
      relations.get(i).writeDelimitedTo(osTrain);
    }
    for(int i = end; i < relations.size(); i ++){
      relations.get(i).writeDelimitedTo(osTrain);
    }
    osTrain.close();
    osTest.close();
  }
  
  private static List<Relation> randomize(List<Relation> relations, int randomSeed) {
    Random rand = new Random(randomSeed);
    Relation [] randomized = new Relation[relations.size()];
    for(int i = 0; i < relations.size(); i ++)
      randomized[i] = relations.get(i);
    for(int j = randomized.length - 1; j > 0; j --){
      int randIndex = rand.nextInt(j);
      Relation tmp = randomized[randIndex];
      randomized[randIndex] = randomized[j];
      randomized[j] = tmp;
    }
    List<Relation> result = new ArrayList<Relation>(Arrays.asList(randomized));
    return result;
  }
}
