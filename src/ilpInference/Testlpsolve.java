package ilpInference;

import java.util.logging.Logger;

import	lpsolve.*;

public class Testlpsolve {
    static Logger logger = Logger.getLogger(Testlpsolve.class.getName());

	public static void main(String[] args) {		
	    logger.info("java.library.path : " + System.getProperty("java.library.path"));
	    logger.info("java.class.path : " + System.getProperty("java.class.path"));

	    try {
	      // Create a problem with 4 variables and 0 constraints
	      LpSolve solver = LpSolve.makeLp(0, 4);

	      // add constraints
	      solver.strAddConstraint("3 2 2 1", LpSolve.LE, 4);
	      solver.strAddConstraint("0 4 3 1", LpSolve.GE, 3);

	      // set objective function
	      solver.strSetObjFn("2 3 -2 3");

	      // solve the problem
	      solver.solve();

	      // print solution
	      System.out.println("Value of objective function: " + solver.getObjective());
	      double[] var = solver.getPtrVariables();
	      for (int i = 0; i < var.length; i++) {
	        System.out.println("Value of var[" + i + "] = " + var[i]);
	      }

	      // delete the problem and free memory
	      solver.deleteLp();
	    }
	    catch (LpSolveException e) {
	       e.printStackTrace();
	    }
	  }
}
