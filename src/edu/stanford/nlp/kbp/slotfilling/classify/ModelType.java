package edu.stanford.nlp.kbp.slotfilling.classify;

public enum ModelType {
  LR_INC, // LR with incomplete information (used at KBP 2011)
  PERCEPTRON, // boring local Perceptron
  PERCEPTRON_INC, // local Perceptron with incomplete negatives
  AT_LEAST_ONCE, // (Hoffman et al, 2011)
  AT_LEAST_ONCE_INC, // AT_LEAST_ONCE with incomplete information
  LOCAL_BAYES, // Mintz++
  JOINT_BAYES, // MIML-RE
  SelPrefOR_EXTRACTOR; // Ajay: NEW Extractor
  
  public static ModelType stringToModel(String s) {
    if (s.equalsIgnoreCase("lr_inc"))
      return ModelType.LR_INC;
    if (s.equalsIgnoreCase("perceptron"))
      return ModelType.PERCEPTRON;
    if(s.equalsIgnoreCase("perceptron_inc"))
      return ModelType.PERCEPTRON_INC;
    if(s.equalsIgnoreCase("atleastonce"))
      return ModelType.AT_LEAST_ONCE;
    if(s.equalsIgnoreCase("atleastonce_inc"))
      return ModelType.AT_LEAST_ONCE_INC;
    if(s.equalsIgnoreCase("localbayes"))
      return ModelType.LOCAL_BAYES;
    if(s.equalsIgnoreCase("jointbayes_inc") || s.equalsIgnoreCase("jointbayes")) 
      return ModelType.JOINT_BAYES;
    if(s.equalsIgnoreCase("selprefor"))
        return ModelType.SelPrefOR_EXTRACTOR;
    throw new RuntimeException("ERROR: Unknown model type: " + s);
  }

}
