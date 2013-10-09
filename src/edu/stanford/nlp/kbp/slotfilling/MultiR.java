package edu.stanford.nlp.kbp.slotfilling;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.zip.GZIPInputStream;

import edu.stanford.nlp.kbp.slotfilling.classify.*;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.ProcessWrapper;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.kbp.slotfilling.multir.ProtobufToMultiLabelDataset;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Triple;

/**
 * Trains and evaluates on the MultiR corpus (Hoffmann et al., 2011)
 * @author Mihai
 *
 */
public class MultiR {
  static final int TUNING_FOLDS = 3;
  
  static class Parameters {
    static final String DEFAULT_TYPE = "jointbayes";
    static final int DEFAULT_FEATURE_COUNT_THRESHOLD = 5;
    static final int DEFAULT_EPOCHS = 15;
    static final int DEFAULT_FOLDS = 5;
    static final String DEFAULT_FILTER = "all";
    static final String DEFAULT_INF_TYPE = "stable";
    static final int DEFAULT_MODEL = 0;
    static final boolean DEFAULT_TRAINY = true;
    
    static final String FOLD_PROP = "fold";

    String trainFile;
    String testFile;
    ModelType type;
    int featureCountThreshold;
    int numberOfTrainEpochs;
    int numberOfFolds;
    String workDir;
    String baseDir;
    String localFilter;
    int featureModel;
    String infType;
    boolean trainY;
    Integer fold;
    
    static Parameters propsToParameters(Properties props) {
      Parameters p = new Parameters();
      p.trainFile = props.getProperty("multir.train");
      p.testFile = props.getProperty("multir.test");
      p.type = ModelType.stringToModel(props.getProperty(
          Props.MODEL_TYPE,  
          DEFAULT_TYPE));
      p.featureCountThreshold = PropertiesUtils.getInt(props, 
          Props.FEATURE_COUNT_THRESHOLD, 
          DEFAULT_FEATURE_COUNT_THRESHOLD);
      p.numberOfTrainEpochs = PropertiesUtils.getInt(props, 
          Props.EPOCHS, 
          DEFAULT_EPOCHS);
      p.numberOfFolds = PropertiesUtils.getInt(props, 
          Props.FOLDS, 
          DEFAULT_FOLDS);
      p.localFilter = props.getProperty(
          Props.FILTER,
          DEFAULT_FILTER);
      p.featureModel = PropertiesUtils.getInt(props, 
          Props.FEATURES,
          DEFAULT_MODEL);
      p.infType = props.getProperty(
          Props.INFERENCE_TYPE,
          DEFAULT_INF_TYPE);
      p.trainY = PropertiesUtils.getBool(props,
          Props.TRAINY,
          DEFAULT_TRAINY);
      p.workDir = props.getProperty(Props.WORK_DIR);
      p.baseDir = props.getProperty(Props.CORPUS_BASE_DIR);
      p.fold = props.containsKey(FOLD_PROP) ?
          Integer.valueOf(props.getProperty(FOLD_PROP)) : null;
      return p;
    }
  }
  
  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    Log.setLevel(Log.stringToLevel(props.getProperty(Props.LOG_LEVEL, "SEVERE")));
    
    if(props.containsKey("tuneFeatures")) {
      tuneFeatures(props);
    } else if(props.containsKey("tuneEpochs")) {
      tuneEpochs(props);
    } else if(props.containsKey("tuneFolds")) {
      tuneFolds(props);
    } else {
      // straight run, train and test
      run(props);
    }
    
  }
  
  private static void tuneFeatures(Properties props) throws Exception {
    Parameters p = Parameters.propsToParameters(props);
    Set<String> sigs = new HashSet<String>();
    int [] values = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    for(int fold = 0; fold < TUNING_FOLDS; fold ++) {
      String trainFile = p.baseDir + File.separator + 
        TUNING_FOLDS + "folds" + File.separator +
        "fold" + fold + File.separator +
        "train.pb.gz";
      String testFile = p.baseDir + File.separator + 
        TUNING_FOLDS + "folds" + File.separator +
        "fold" + fold + File.separator +
        "test.pb.gz";
      p.trainFile = trainFile;
      p.testFile = testFile;
      p.fold = fold;
      for(int v: values) {
        Log.severe("Launching job for fold #" + fold + " with value " + v + "...");
        p.featureCountThreshold = v;
        launchJob(p, trainFile, testFile);
        String sig = makeSignature(p);
        sigs.add(sig);
      }
    }
    
    waitFor(sigs, p);
    reportResults(sigs, p);
  }
  
  private static void tuneEpochs(Properties props) throws Exception {
    Parameters p = Parameters.propsToParameters(props);
    Set<String> sigs = new HashSet<String>();
    int [] values = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    for(int fold = 0; fold < TUNING_FOLDS; fold ++) {
      String trainFile = p.baseDir + File.separator + 
        TUNING_FOLDS + "folds" + File.separator +
        "fold" + fold + File.separator +
        "train.pb.gz";
      String testFile = p.baseDir + File.separator + 
        TUNING_FOLDS + "folds" + File.separator +
        "fold" + fold + File.separator +
        "test.pb.gz";
      p.trainFile = trainFile;
      p.testFile = testFile;
      p.fold = fold;
      for(int v: values) {
        Log.severe("Launching job for fold #" + fold + " with value " + v + "...");
        p.numberOfTrainEpochs = v;
        launchJob(p, trainFile, testFile);
        //launchSequential(p, trainFile, testFile);
        String sig = makeSignature(p);
        sigs.add(sig);
      }
    }
    
    waitFor(sigs, p);
    reportResults(sigs, p);
  }
  
  private static void tuneFolds(Properties props) throws Exception {
    Parameters p = Parameters.propsToParameters(props);
    Set<String> sigs = new HashSet<String>();
    int [] values = { 2, 3, 5, 6 };
    for(int fold = 0; fold < TUNING_FOLDS; fold ++) {
      String trainFile = p.baseDir + File.separator + 
        TUNING_FOLDS + "folds" + File.separator +
        "fold" + fold + File.separator +
        "train.pb.gz";
      String testFile = p.baseDir + File.separator + 
        TUNING_FOLDS + "folds" + File.separator +
        "fold" + fold + File.separator +
        "test.pb.gz";
      p.trainFile = trainFile;
      p.testFile = testFile;
      p.fold = fold;
      for(int v: values) {
        Log.severe("Launching job for fold #" + fold + " with value " + v + "...");
        p.numberOfFolds = v;
        launchJob(p, trainFile, testFile);
        String sig = makeSignature(p);
        sigs.add(sig);
      }
    }
    
    waitFor(sigs, p);
    reportResults(sigs, p);
  }
  
  static class Result {
    double p;
    double r;
    double f1;
    String sig;
  }
  
  private static void reportResults(Set<String> sigs, Parameters p) {
    List<Result> results = new ArrayList<Result>();
    for(String sig: sigs) {
      try {
        BufferedReader is = new BufferedReader(new FileReader(p.workDir + File.separator + sig + ".score"));
        String line = is.readLine();
        String [] bits = line.split("\\s+");
        Result r = new Result();
        r.sig = sig;
        r.p = Double.valueOf(bits[1]);
        r.r = Double.valueOf(bits[3]);
        r.f1 = Double.valueOf(bits[5]);
        results.add(r);
        is.close();
      } catch(IOException e) {
        System.err.println("WARNING: cannot read results for sig: " + sig);
      }
    }
    
    results = averageFolds(results);
    
    Collections.sort(results, new Comparator<Result>() {
      @Override
      public int compare(Result o1, Result o2) {
        if(o1.f1 > o2.f1) return -1;
        if(o1.f1 == o2.f1) return 0;
        return 1;
      }}); 
    
    for(Result r: results) {
      System.out.println(r.sig + "\tP " + r.p + " R " + r.r + " F1 " + r.f1);
    }
  }
  
  private static List<Result> averageFolds(List<Result> foldResults) {
    Map<String, List<Result>> resultsBySig = new HashMap<String, List<Result>>();
    for(Result r: foldResults) {
      int end = r.sig.indexOf("_fold");
      assert(end > 0);
      String sig = r.sig.substring(0, end);
      List<Result> bySig = resultsBySig.get(sig);
      if(bySig == null) {
        bySig = new ArrayList<Result>();
        resultsBySig.put(sig, bySig);
      }
      bySig.add(r);
    }
    
    List<Result> results = new ArrayList<Result>();
    for(String sig: resultsBySig.keySet()) {
      Result r = new Result();
      r.sig = sig;
      List<Result> bySig = resultsBySig.get(sig);
      r.p = sumP(bySig) / bySig.size();
      r.r = sumR(bySig) / bySig.size();
      r.f1 = sumF1(bySig) / bySig.size();
      results.add(r);
    }
    
    return results;
  }
  
  private static double sumP(List<Result> rs) {
    double sum = 0;
    for(Result r: rs) sum += r.p;
    return sum;
  }
  private static double sumR(List<Result> rs) {
    double sum = 0;
    for(Result r: rs) sum += r.r;
    return sum;
  }
  private static double sumF1(List<Result> rs) {
    double sum = 0;
    for(Result r: rs) sum += r.f1;
    return sum;
  }
  
  private static void waitFor(Set<String> sigs, Parameters p) throws InterruptedException {
    Log.severe("Waiting for " + sigs.size() + " jobs.");
    int remaining = 0;
    while(true) {
      Set<String> inc = countCompleted(sigs, p.workDir);
      if(inc.size() == 0) {
        Log.severe("All jobs complete!");
        break;
      } else {
        if(inc.size() != remaining)
          Log.severe("Still waiting for " + inc.size() + " jobs: " + inc);
        remaining = inc.size();
      }
      Thread.sleep(60000);
    }
  }
  
  private static Set<String> countCompleted(Set<String> sigs, String workDir) {
    Set<String> incomplete = new HashSet<String>();
    for(String sig: sigs) {
      File f = new File(workDir + File.separator + sig + ".score");
      if(! f.exists()) incomplete.add(sig);
    }
    return incomplete;
  }
  
  static boolean launchSequential(Parameters p, String trainFile, String testFile) throws Exception {
    Properties props = new Properties();
    props.setProperty(Props.WORK_DIR, p.workDir);
    props.setProperty("multir.train", p.trainFile);
    props.setProperty("multir.test", p.testFile);
    props.setProperty(Props.FEATURE_COUNT_THRESHOLD, Integer.toString(p.featureCountThreshold));
    props.setProperty(Props.EPOCHS, Integer.toString(p.numberOfTrainEpochs));
    props.setProperty(Props.FOLDS, Integer.toString(p.numberOfFolds));
    props.setProperty(Props.FILTER, p.localFilter);
    props.setProperty(Props.INFERENCE_TYPE, p.infType);
    props.setProperty(Props.FEATURES, Integer.toString(p.featureModel));
    if(p.fold != null)
      props.setProperty(Parameters.FOLD_PROP, Integer.toString(p.fold));
    run(props);
    return true;
  }
  
  private static final String TRAINER_MEMORY = "12g";

  private static boolean launchJob(Parameters p, String trainFile, String testFile) throws Exception {
    String sig = makeSignature(p);
    File logDir = new File(p.workDir + File.separator + sig);
    
    // check if the job is running or has run
    if(logDir.exists()){
      Log.severe("Job " + sig + " already done. Its log directory exists! Remove it to rerun.");
      return true;
    }
    
    mkDir(logDir);
    
    String cmd = "nlpsub -v --join-output-streams --queue=long --debug --cores=2"
      + " --log-dir=" + logDir.getAbsolutePath() + " --name=" + sig 
      + " java -ea -Xmx" + TRAINER_MEMORY + " -XX:MaxPermSize=512m" 
      + " edu.stanford.nlp.kbp.slotfilling.MultiR" 
      + " -" + Props.WORK_DIR + " " + p.workDir 
      + " -multir.train " + p.trainFile 
      + " -multir.test " + p.testFile 
      + " -" + Props.FEATURE_COUNT_THRESHOLD + " " + p.featureCountThreshold
      + " -" + Props.EPOCHS + " " + p.numberOfTrainEpochs
      + " -" + Props.FOLDS + " " + p.numberOfFolds 
      + " -" + Props.FILTER + " " + p.localFilter
      + " -" + Props.INFERENCE_TYPE + " " + p.infType
      + " -" + Props.FEATURES + " " + p.featureModel
      + " -" + Props.TRAINY + " " + p.trainY
      + (p.fold != null ? " -" + Parameters.FOLD_PROP + " " + p.fold : "")
      + " -run"; 
    Log.severe("Command: " + cmd);
    
    return launch(cmd);
  }
  
  private static boolean launch(String cmd) throws IOException, InterruptedException {
    ProcessWrapper process = ProcessWrapper.create(cmd);
    process.waitFor();
    int exitCode = process.exitValue();
    if (exitCode == 0) {
      Log.severe("The above command terminated successfully.");
    } else {
      Log.severe("The above command exited with code: " + exitCode);
    }

    String err = process.consumeErrorStream();
    if (err.length() > 0) {
      Log.severe("Error stream contained:\n" + err);
    }
    String stdout = process.consumeReadStream();
    if (stdout.length() > 0) {
      Log.severe("Stdout stream contained:\n" + stdout);
    }

    return (exitCode == 0);
  }
  
  private static void mkDir(File dir) throws InterruptedException {
    if (!dir.exists()) {
      dir.mkdirs();
    }
    Thread.sleep(5000); // let NFS catch up
    assert (dir.exists() && dir.isDirectory());
  }
   
  private static void run(Properties props) throws Exception {
    Parameters p = Parameters.propsToParameters(props);
    String sig = makeSignature(p);
    Log.severe("Using signature: " + sig);
    String modelPath = p.workDir + File.separator + sig + ".ser";
    String scoreFile = p.workDir + File.separator + sig + ".score";
    boolean showPRCurve = PropertiesUtils.getBool(props, Props.SHOW_CURVE, true);

    List<Set<String>> goldLabels = new ArrayList<Set<String>>();
    List<Counter<String>> predictedLabels = new ArrayList<Counter<String>>();
    Triple<Double, Double, Double> score = run(p, modelPath, goldLabels, predictedLabels);
    System.out.println("P " + score.first() + " R " + score.second() + " F1 " + score.third());
    
    PrintStream os = new PrintStream(new FileOutputStream(scoreFile));
    os.println("P " + score.first() + " R " + score.second() + " F1 " + score.third());
    os.close();
    
    if(showPRCurve) {
      String curveFile = p.workDir + File.separator + sig + ".curve";
      os = new PrintStream(new FileOutputStream(curveFile));
      // generatePRCurve(os, goldLabels, predictedLabels);
      generatePRCurveNonProbScores(os, goldLabels, predictedLabels);
      os.close();
      System.out.println("P/R curve values saved in file " + curveFile);
    }
  }

  private static void generatePRCurveNonProbScores(PrintStream os,
                                                   List<Set<String>> goldLabels,
                                                   List<Counter<String>> predictedLabels) {
    // each triple stores: position of tuple in gold, one label for this tuple, its score
    List<Triple<Integer, String, Double>> preds = convertToSorted(predictedLabels);
    double prevP = -1, prevR = -1;
    int START_OFFSET = 10; // score at least this many predictions (makes no sense to score 1...)
    for(int i = START_OFFSET; i < preds.size(); i ++) {
      List<Triple<Integer, String, Double>> filteredLabels = preds.subList(0, i);
      Triple<Double, Double, Double> score = score(filteredLabels, goldLabels);
      if(score.first() != prevP || score.second() != prevR) {
        double ratio = (double) i / (double) preds.size();
        os.println(ratio + " P " + score.first() + " R " + score.second() + " F1 " + score.third());
        prevP = score.first();
        prevR = score.second();
      }
    }
  }
  private static List<Triple<Integer, String, Double>> convertToSorted(List<Counter<String>> predictedLabels) {
    List<Triple<Integer, String, Double>> sorted = new ArrayList<Triple<Integer, String, Double>>();
    for(int i = 0; i < predictedLabels.size(); i ++) {
      for(String l: predictedLabels.get(i).keySet()) {
        double s = predictedLabels.get(i).getCount(l);
        sorted.add(new Triple<Integer, String, Double>(i, l, s));
      }
    }
    Collections.sort(sorted, new Comparator<Triple<Integer, String, Double>>() {
      @Override
      public int compare(Triple<Integer, String, Double> t1, Triple<Integer, String, Double> t2) {
        if(t1.third() > t2.third()) return -1;
        else if(t1.third() < t2.third()) return 1;
        return 0;
      }
    });
    return sorted;
  }
  private static Triple<Double, Double, Double> score(List<Triple<Integer, String, Double>> preds, List<Set<String>> golds) {
    int total = 0, predicted = 0, correct = 0;
    for(int i = 0; i < golds.size(); i ++) {
      Set<String> gold = golds.get(i);
      total += gold.size();
    }
    for(Triple<Integer, String, Double> pred: preds) {
      predicted ++;
      if(golds.get(pred.first()).contains(pred.second()))
        correct ++;
    }

    double p = (double) correct / (double) predicted;
    double r = (double) correct / (double) total;
    double f1 = (p != 0 && r != 0 ? 2*p*r/(p+r) : 0);
    return new Triple<Double, Double, Double>(p, r, f1);
  }

  private static void generatePRCurve(PrintStream os,
      List<Set<String>> goldLabels, 
      List<Counter<String>> predictedLabels) {
    for(double t = 1.0; t >= 0; ) {
      List<Counter<String>> filteredLabels = keepAboveThreshold(predictedLabels, t);
      Triple<Double, Double, Double> score = JointlyTrainedRelationExtractor.score(goldLabels, filteredLabels);
      os.println(t + " P " + score.first() + " R " + score.second() + " F1 " + score.third());
      if(t > 1.0) t -= 1.0;
      else if(t > 0.99) t -= 0.0001;
      else if(t > 0.95) t -= 0.001;
      else t -= 0.01;
    }
  }
  
  private static List<Counter<String>> keepAboveThreshold(List<Counter<String>> labels, double threshold) {
    List<Counter<String>> filtered = new ArrayList<Counter<String>>();
    for(Counter<String> group: labels) {
      Counter<String> filteredGroup = new ClassicCounter<String>();
      for(String l: group.keySet()) {
        double v = group.getCount(l);
        if(v >= threshold) filteredGroup.setCount(l, v);
      }
      filtered.add(filteredGroup);
    }
    return filtered;
  }
  
  private static Triple<Double, Double, Double> run(
      Parameters p,
      String modelPath,
      List<Set<String>> goldLabels,
      List<Counter<String>> predictedLabels) throws IOException, ClassNotFoundException {
    JointlyTrainedRelationExtractor extractor = null;
    if(p.type == ModelType.JOINT_BAYES) {
      String initialModelPath = modelPath.replaceAll("\\.ser", ".initial.ser");
      JointBayesRelationExtractor ex = new JointBayesRelationExtractor(
          initialModelPath, 
          p.numberOfTrainEpochs, 
          p.numberOfFolds, 
          p.localFilter, 
          p.featureModel,
          p.infType,
          p.trainY,
          false);
      ex.setSerializedModelPath(modelPath);
      extractor = ex;
    } else if(p.type == ModelType.LOCAL_BAYES) {
      extractor = new JointBayesRelationExtractor(
          null, 
          p.numberOfTrainEpochs, 
          p.numberOfFolds, 
          p.localFilter, 
          p.featureModel,
          p.infType,
          p.trainY,
          true);
    } else if(p.type == ModelType.AT_LEAST_ONCE) {
      extractor = new HoffmannExtractor(p.numberOfTrainEpochs);
    } else {
      throw new RuntimeException("ERROR: unsupported model type: " + p.type);
    }
    
    if(new File(modelPath).exists()) {
      // load an existing model
      Log.severe("Existing model found at " + modelPath + ". Will NOT train a new one.");
      ObjectInputStream in = new ObjectInputStream(new FileInputStream(modelPath));
      extractor.load(in);
      in.close();
    } else {
      // train a new model
      InputStream is = new GZIPInputStream(
          new BufferedInputStream(new FileInputStream(p.trainFile)));
      MultiLabelDataset<String, String> trainDataset =  
        ProtobufToMultiLabelDataset.toMultiLabelDataset(is);
      is.close();
      trainDataset.randomize(1);
      trainDataset.applyFeatureCountThreshold(p.featureCountThreshold);
      extractor.train(trainDataset);

      // save
      extractor.save(modelPath);
    }
    
    // test
    List<List<Collection<String>>> relations = 
      new ArrayList<List<Collection<String>>>();
    InputStream is = new GZIPInputStream(
        new BufferedInputStream(new FileInputStream(p.testFile)));
    ProtobufToMultiLabelDataset.toDatums(is, relations, goldLabels);
    is.close();
    Triple<Double, Double, Double> score = extractor.test(relations, goldLabels, predictedLabels);
    //Triple<Double, Double, Double> score = extractor.oracle(relations, goldLabels, predictedLabels);
    return score;
  }
  
  private static String makeSignature(Parameters p) {
    StringBuffer os = new StringBuffer();
    os.append("multir");
    os.append("_" + p.type);
    os.append("_T" + p.featureCountThreshold);
    os.append("_E" + p.numberOfTrainEpochs);
    os.append("_NF" + p.numberOfFolds);
    os.append("_F" + p.localFilter);
    os.append("_M" + p.featureModel);
    os.append("_I" + p.infType);
    os.append("_Y" + p.trainY);

    // in case of cross-validation tuning
    if(p.fold != null) 
      os.append("_fold" + p.fold);
    
    return os.toString();
  }
}
