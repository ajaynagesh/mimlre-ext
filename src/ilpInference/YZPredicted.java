package ilpInference;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;

public class YZPredicted {
	  Counter<Integer> yPredicted;
	  int [] zPredicted;
	  Counter<Integer> yPredictedScores;
	  public YZPredicted(int sz){
		  yPredicted = new ClassicCounter<Integer>();
		  zPredicted = new int [sz];
		  yPredictedScores = new ClassicCounter<Integer>();
	  }
	  public Counter<Integer> getYPredicted(){
		  return yPredicted;
	  }
	  public int [] getZPredicted(){
		  return zPredicted;
	  }
	  public Counter<Integer> getYPredictedScores(){
		  return yPredictedScores;
	  }
}
