package ilpInference;

import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;

public class YZPredicted {
	  Counter<Integer> yPredicted;
	  int [] zPredicted;
	  public YZPredicted(int sz){
		  yPredicted = new ClassicCounter<Integer>();
		  zPredicted = new int [sz];
	  }
	  public Counter<Integer> getYPredicted(){
		  return yPredicted;
	  }
	  public int [] getZPredicted(){
		  return zPredicted;
	  }
}
