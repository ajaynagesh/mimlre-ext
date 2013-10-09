package edu.stanford.nlp.wikipedia.pipeline;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.pipeline.ChunkAnnotationUtils;
import edu.stanford.nlp.process.AbstractTokenizer;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Does post processing on tokens to fix up stuff we don't like
 * while maintaining the original character offsets
 *  (copied from NFLTokenizer)
 * Note: it is crucial to maintain the character offsets of the original tokens 
 *       because they are used for aligning the annotations against text
 * @author Angel Chang
 */
public class TokenizerPostProcessorAnnotator implements Annotator {
  private PostProcessingTokenizer tokenizer;

  public TokenizerPostProcessorAnnotator() {
    this.tokenizer = new PostProcessingTokenizer();
  }

  public void annotate(Annotation annotation) {
    if (annotation.has(CoreAnnotations.TokensAnnotation.class)) {
      List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
      tokens = tokenizer.postprocess(tokens);
      annotation.set(CoreAnnotations.TokensAnnotation.class, tokens);
    } else if (annotation.has(CoreAnnotations.TextAnnotation.class)) {
      PostProcessingTokenizer strTokenizer = new PostProcessingTokenizer(annotation.get(CoreAnnotations.TextAnnotation.class));
      List<CoreLabel> tokens = strTokenizer.tokenize();
      annotation.set(CoreAnnotations.TokensAnnotation.class, tokens);
    } else {
      throw new RuntimeException("unable to find tokens or text in annotation: " + annotation);
    }
  }

  public static class PostProcessingTokenizer  {
    AbstractTokenizer<CoreLabel> tokenizer;
    CoreLabelTokenFactory tokenFactory = new CoreLabelTokenFactory();

    public PostProcessingTokenizer(String buffer) {
      StringReader sr = new StringReader(buffer);
      String options = "invertible,ptb3Escaping=true";
      tokenizer = new PTBTokenizer<CoreLabel>(sr, tokenFactory, options);
    }

    public PostProcessingTokenizer() {
      tokenizer = null;
    }

    public List<CoreLabel> tokenize() {
      List<CoreLabel> tokens = tokenizer.tokenize();
      return postprocess(tokens);
    }

    public List<CoreLabel> postprocess(List<CoreLabel> tokens) {
      tokens = breakDashes(tokens);
      return tokens;
    }

    private static final Pattern ANYDASH_PATTERN = Pattern.compile("\\w+\\s*(-)\\s*\\w+");

    /** Do not break words at dashes if the prefix is in this set */
    private static final HashSet<String> VALID_PREFIXES = new HashSet<String>(Arrays.asList(new String[]{}));

    /**
     * Separate tokens that look like "stuff-stuff" . These may include relevant information.
     * @param tokens
     */
    private List<CoreLabel> breakDashes(List<CoreLabel> tokens) {
      List<CoreLabel> output = new ArrayList<CoreLabel>();
      for(int i = 0; i < tokens.size(); i ++){
        CoreLabel t = tokens.get(i);
        Matcher m = ANYDASH_PATTERN.matcher(t.word());
        if(m.find()){
          int dashPos = m.start(1);
          String s1 = t.word().substring(0, dashPos);
          if(VALID_PREFIXES.contains(s1)){
            output.add(t);
          } else {
            // TODO: Fix our old stuff, so we don't need to copy these unset annotation
            CoreLabel t1 = tokenFactory.makeToken(s1, t.beginPosition(), dashPos);
            ChunkAnnotationUtils.copyUnsetAnnotations(t, t1);
            output.add(t1);
            String s2 = "-";
            CoreLabel t2 = tokenFactory.makeToken(s2, t.beginPosition() + dashPos, 1);
            ChunkAnnotationUtils.copyUnsetAnnotations(t, t2);
            output.add(t2);
            String s3 = t.word().substring(dashPos + 1);
            CoreLabel t3 = tokenFactory.makeToken(s3, t.beginPosition() + dashPos + 1, t.endPosition() - t.beginPosition() - dashPos - 1);
            ChunkAnnotationUtils.copyUnsetAnnotations(t, t3);
            output.add(t3);
          }
        } else {
          output.add(t);
        }
      }
      return output;
    }
  }

}
