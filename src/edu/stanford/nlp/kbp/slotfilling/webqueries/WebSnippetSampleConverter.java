package edu.stanford.nlp.kbp.slotfilling.webqueries;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.util.CollectionFactory;
import edu.stanford.nlp.util.CollectionValuedMap;

import java.io.*;
import java.util.*;

/**
 * Converts web snippet to column formatted data
 * Columns are
 *   Label (+,-)
 *   SlotName - Slotname for which this sample should be used (only if outputting to one big file)
 *   SnippetQueryType - QueryType from which the snippet came from
 *   SnippetSlotName - SlotName from which the snippet came from
 *   Entity - Entity string
 *   SlotValue - Filler of slot
 *   Text - Snippet text
 * @author Angel Chang
 */
public class WebSnippetSampleConverter extends WebSnippetProcessor.AbstractWebSnippetHandler {
  char delimiter = '\t';
  boolean oneFilePerSlot = true;
  boolean posSamplesOnly = true;
  String sampleName = "snippets";
  String sampleDir;
  Set<RelationMentionSnippets.QueryType> matchQueryTypes;

  private Map<String,PrintWriter> cachedWriters;
  private static final String POS_LABEL = "+";
  private static final String NEG_LABEL = "-";
  private WebSnippetMasker textMasker;

  // Only used if we need to do negsamples
  private CollectionValuedMap<String,String> slotSamples;

  @Override
  public void init(Properties props) {
    super.init(props);
    textMasker = new WebSnippetMasker(props);
    cachedWriters = new HashMap<String,PrintWriter>();
    sampleDir = props.getProperty("kbp.websnippets.samples.dir");
    sampleName = props.getProperty("kbp.websnippets.samples.name", sampleName);
    posSamplesOnly = Boolean.parseBoolean(props.getProperty("kbp.websnippets.samples.posSamplesOnly", "true"));
    oneFilePerSlot = Boolean.parseBoolean(props.getProperty("kbp.websnippets.samples.oneFilePerSlot", "true"));
    String queryTypesStr = props.getProperty("kbp.websnippets.samples.queryTypes", RelationMentionSnippets.QueryType.EE.name());
    String[] queryTypes = queryTypesStr.split(",");
    matchQueryTypes = EnumSet.noneOf(RelationMentionSnippets.QueryType.class);
    for (String queryType:queryTypes) {
      queryType = queryType.trim();
      matchQueryTypes.add(RelationMentionSnippets.QueryType.valueOf(queryType));
    }
    try {
      createSampleDir();
    } catch (IOException ex) {
      throw new RuntimeException(ex);
    }
    if (!posSamplesOnly) {
      slotSamples = new CollectionValuedMap<String,String>(CollectionFactory.<String>arrayListFactory());
    }
  }

  @Override
  public void finish() {
    super.finish();
    if (!posSamplesOnly) {
      addNegSamples();
    }
    close();
  }

  public void close()
  {
    if (cachedWriters != null) {
      for (PrintWriter pw: cachedWriters.values()) {
        pw.close();
      }
      cachedWriters.clear();
    }
  }

  public String getOutputFilename(String outputName)
  {
    return sampleDir + "/" + outputName + ".samples.gz";
  }

  public String getOutputName(RelationMentionSnippets snippets)
  {
    return getOutputName(snippets.getSlotName());
  }

  public String getOutputName(String slotName)
  {
    if (oneFilePerSlot) {
      return slotName;
    } else {
      return sampleName;
    }
  }

  public PrintWriter getPrintWriter(String outputName)
  {
    PrintWriter pw = cachedWriters.get(outputName);
    if (pw == null) {
      try {
        String filename = getOutputFilename(outputName);
        pw = IOUtils.getPrintWriter(filename);
      } catch (IOException ex) {
        throw new RuntimeException(ex);
      }
      cachedWriters.put(outputName, pw);
    }
    return pw;
  }

  public void createSampleDir() throws IOException
  {
    File dir = new File(sampleDir);
    if (!dir.exists()) {
      dir.mkdirs();
    } else if (!dir.isDirectory()) {
      throw new IOException("WARNING: " + sampleDir + " is not a directory");
    }
  }

  private final Random rand = new Random();
  public Collection<String> getSamples(Set<String> slots, int nSamples)
  {
    rand.setSeed(System.currentTimeMillis());
    List<String> selected = new ArrayList<String>(nSamples);
    int n = 0;
    for (String slot:slots) {
      Collection<String> s = slotSamples.get(slot);
      for (String sample:s) {
        n++;
        if (selected.size() >= nSamples) {
          // Update selection
          double threshold = ((double) nSamples)/n;
          double v = rand.nextDouble();
          if (v < threshold) {
            int i = rand.nextInt(nSamples);
            selected.set(i, sample);
          }
        } else {
          selected.add(sample);
        }
      }
    }
    return selected;
  }

  public void addNegSamples(String slot, Set<String> others)
  {
    int nSamples = slotSamples.get(slot).size();
    Collection<String> samples = getSamples(others, nSamples);
    for (String sample:samples) {
      outputSample(slot, sample, NEG_LABEL);
    }
  }

  public void addNegSamples()
  {
    Set<String> perSlots = new HashSet<String>();
    Set<String> orgSlots = new HashSet<String>();
    Set<String> allSlots = slotSamples.keySet();
    for (String slot:allSlots) {
      if (slot.startsWith("per:")) {
        perSlots.add(slot);
      } else if (slot.startsWith("org:")) {
        orgSlots.add(slot);
      } else {
        System.err.println("WARNING: Skipping unknown slot type " + slot);
      }
    }
    for (String slot:perSlots) {
      addNegSamples(slot, orgSlots);
    }
    for (String slot:orgSlots) {
      addNegSamples(slot, perSlots);
    }
  }

  public void outputSample(RelationMentionSnippets snippets, String sampleString, String label)
  {
    outputSample(snippets.slotName, sampleString, label);
  }

  public void outputSample(String slotName, String sampleString, String label)
  {
    // Output sample as column formatted data
    String outputName = getOutputName(slotName);
    PrintWriter pw = getPrintWriter(outputName);
    pw.print(label);
    pw.print(delimiter);
    if (!oneFilePerSlot) {
      pw.print(slotName);
      pw.print(delimiter);
    }
    pw.println(sampleString);
  }

  public String getSampleString(RelationMentionSnippets snippets, WebSnippet snippet)
  {
    StringBuilder sb = new StringBuilder();
    sb.append(snippets.getQueryType()).append(delimiter);
    sb.append(snippets.getSlotName()).append(delimiter);
    sb.append(snippets.getEntityName()).append(delimiter);
    sb.append(snippets.getSlotValue()).append(delimiter);
    String text = snippet.getText();
    sb.append(text);
    if (textMasker != null) {
      String maskedText = textMasker.getMaskedText(text, snippets.getEntityName(), snippets.getSlotValue());
      sb.append(delimiter).append(maskedText);
    }
    return sb.toString();
  }

  public void processSnippet(RelationMentionSnippets snippets, WebSnippet snippet) {
    if (matchQueryTypes == null || matchQueryTypes.contains(snippets.getQueryType())) {
      String sampleString = getSampleString(snippets, snippet);
      outputSample(snippets, sampleString, POS_LABEL);
      if (!posSamplesOnly) {
        slotSamples.add(snippets.getSlotName(), sampleString);
      }
    }
  }

}
