package ilpInference;

import java.util.List;

import net.sf.javailp.Constraint;
import net.sf.javailp.Linear;
import net.sf.javailp.OptType;
import net.sf.javailp.Problem;
import net.sf.javailp.Result;
import net.sf.javailp.Solver;
import net.sf.javailp.SolverFactory;
import net.sf.javailp.SolverFactoryLpSolve;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Index;

public class InferenceWrappers {
	
	public Counter<Integer> generateYPredictedILP(List<Counter<Integer>> scores, 
												  int numOfMentions, 
												  Index<String> yLabelIndex, 
												  Counter<Integer> typeBiasScores){
		
		Counter<Integer> yPredicted = new ClassicCounter<Integer>();
		System.out.println("Calling ILP inference");
		
		int numConstraints = 0;
		
		SolverFactory factory = new SolverFactoryLpSolve();
		factory.setParameter(Solver.VERBOSE, 0);
		factory.setParameter(Solver.TIMEOUT, 100); // set timeout to 100 seconds

		Problem problem = new Problem();
		
		Linear objective = new Linear();
		
		////////////// Objective --------------------------------------
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			Counter<Integer> score = scores.get(mentionIdx);
			for(int label : score.keySet()){
				String var = "z"+mentionIdx+"_"+"y"+label;
				double coeff = score.getCount(label);
				objective.add(coeff, var);
				
				//System.out.print(score.getCount(label) + "  " + "z"+mentionIdx+"_"+"y"+label + " + ");
			}
		}
		//System.out.println();
		for(String yLabel : yLabelIndex){
			int y = yLabelIndex.indexOf(yLabel);
			String var = "y"+y;
			double coeff = typeBiasScores.getCount(y);
			objective.add(coeff, var);
			
			//System.out.print(typeBiasScores.getCount(y)+" y"+y+" + ");
		}
		
		problem.setObjective(objective, OptType.MAX);
		/////////// -----------------------------------------------------
		
		//System.out.println("\n-----------------");
		/////////// Constraints ------------------------------------------

		/// 1. equality constraints \Sum_i z_ji = 1 \forall j
		Linear constraint = new Linear();
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			for(String yLabel : yLabelIndex){
				int y = yLabelIndex.indexOf(yLabel);
				String var = "z"+mentionIdx+"_"+"y"+y;
				constraint.add(1, var);
				
				//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");				
			}
			
			problem.add(constraint, "=", 1);
			numConstraints++;
			//System.out.println(" 0 = "+ "1");
		}
		
		//System.out.println("\n-----------------");
		
		/// 2. inequality constraint -- 1 ... z_ji <= y_i \forall j,i
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			constraint = new Linear();
			for(String yLabel : yLabelIndex){
				int y = yLabelIndex.indexOf(yLabel);
				String var1 = "z"+mentionIdx+"_"+"y"+y;
				String var2 = "y"+y;
				constraint.add(1, var1);
				constraint.add(-1, var2);
				problem.add(constraint, "<=", 0);
				numConstraints++;
				//System.out.println("z"+mentionIdx+"_"+"y"+y +" - " + "y"+y + " <= 0");
			}
		}
		
		//System.out.println("\n-----------------");
		/// 3. inequality constraint -- 2 ... y_i <= \Sum_j z_ji \forall i
		/////////// ------------------------------------------------------
		for(String yLabel : yLabelIndex){
			constraint = new Linear();
			int y = yLabelIndex.indexOf(yLabel);
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				String var = "z"+mentionIdx+"_"+"y"+y;
				constraint.add(1, var);
				//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");
			}
			constraint.add(-1, "y"+y);
			problem.add(constraint, ">=", 0);
			numConstraints++;
			//System.out.println(" 0 - " + "y"+y +" >= 0" );
		}
		
		// Set the types of all variables to Binary
		for(Object var : problem.getVariables())
			problem.setVarType(var, Boolean.class);
		
//		System.out.println("Num of variables : " + problem.getVariablesCount());
//		System.out.println("Num of Constraints : " + problem.getConstraintsCount());
//		System.out.println("Objective Function : ");
//		System.out.println(problem.getObjective());
//		System.out.println("Constraints : ");
//		for(Constraint c : problem.getConstraints())
//			System.out.println(c);
		
		// Solve the ILP problem by calling the ILP solver
		Solver solver = factory.get();
		Result result = solver.solve(problem);
		
		System.out.println("Result : " + result);
		
		System.exit(0);
		return yPredicted;
	}

}
