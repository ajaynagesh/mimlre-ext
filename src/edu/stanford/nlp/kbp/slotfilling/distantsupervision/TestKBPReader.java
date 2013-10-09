package edu.stanford.nlp.kbp.slotfilling.distantsupervision;

import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Properties;
import java.util.logging.Level;

import edu.stanford.nlp.kbp.slotfilling.FeatureFactory;
import edu.stanford.nlp.kbp.slotfilling.common.Log;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.StringUtils;

public class TestKBPReader {
  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    Log.setLevel(Level.SEVERE);
    KBPReader reader = new KBPReader(props, false, PropertiesUtils.getBool(props, Props.KBP_TEMPORAL, false), PropertiesUtils.getBool(props, Props.KBP_DIAGNOSTICMODE, false));
    reader.setLoggerLevel(Level.SEVERE);
    assert (props.getProperty("test.input") != null);

    PrintStream os = new PrintStream(new FileOutputStream(props.getProperty("test.output", "test_datums.dat")));
    Counter<String> labelStats = new ClassicCounter<String>();
    Counter<String> domainStats = new ClassicCounter<String>();
    String[] relationFeatures = props.getProperty(Props.RELATION_FEATS).split(",\\s*");
    FeatureFactory rff = new FeatureFactory(relationFeatures);
    rff.setDoNotLexicalizeFirstArgument(true);
    reader.parse(os, props.getProperty("test.input"), rff, labelStats, domainStats);
    os.close();
    System.err.println("LABEL STATS: " + labelStats);
    System.err.println("DOMAIN STATS: " + domainStats);
  }
}
