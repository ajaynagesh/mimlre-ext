package ilpInference;

import java.util.Map;
import java.util.logging.Logger;

import net.sf.javailp.Linear;
import net.sf.javailp.OptType;
import net.sf.javailp.Problem;
import net.sf.javailp.Result;
import net.sf.javailp.Solver;
import net.sf.javailp.SolverFactory;
import net.sf.javailp.SolverFactoryGLPK;
import net.sf.javailp.SolverFactoryLpSolve;

public class TestJavaILP {
    static Logger logger = Logger.getLogger(TestJavaILP.class.getName());
	public static void main(String args[]){
	    //logger.info("java.library.path : " + System.getProperty("java.library.path"));
	    //logger.info("java.class.path : " + System.getProperty("java.class.path"));
		SolverFactory factory = new SolverFactoryLpSolve(); // use lp_solve
		Map<Object, Object> params = factory.getParameters();
		System.out.println("PARAMS : " + params);
		factory.setParameter(Solver.VERBOSE, 0);
		factory.setParameter(Solver.TIMEOUT, 100); // set timeout to 100 seconds
		System.out.println("PARAMS after: " + params);

		
		/**
		* Constructing a Problem: 
		* Maximize: 143x+60y 
		* Subject to: 
		* 120x+210y <= 15000 
		* 110x+30y <= 4000 
		* x+y <= 75
		* 
		* With x,y being integers
		* 
		*/
		Problem problem = new Problem();

		Linear linear = new Linear();
		linear.add(143, "x");
		linear.add(60, "y");

		problem.setObjective(linear, OptType.MAX);

		linear = new Linear();
		linear.add(120, "x");
		linear.add(210, "y");

		problem.add(linear, "<=", 15000);

		linear = new Linear();
		linear.add(110, "x");
		linear.add(30, "y");

		problem.add(linear, "<=", 4000);

		linear = new Linear();
		linear.add(1, "x");
		linear.add(1, "y");

		problem.add(linear, "<=", 75);

		problem.setVarType("x", Integer.class);
		problem.setVarType("y", Integer.class);

		Solver solver = factory.get(); // you should use this solver only once for one problem
		Result result = solver.solve(problem);

		System.out.println(result);

		/**
		* Extend the problem with x <= 16 and solve it again
		*/
		problem.setVarUpperBound("x", 16);

		solver = factory.get();
		result = solver.solve(problem);

		System.out.println(result);
	}
}
