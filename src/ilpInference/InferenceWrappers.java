package ilpInference;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import net.sf.javailp.Constraint;
import net.sf.javailp.Linear;
import net.sf.javailp.OptType;
import net.sf.javailp.Problem;
import net.sf.javailp.Result;
import net.sf.javailp.Solver;
import net.sf.javailp.SolverFactory;
import net.sf.javailp.SolverFactoryLpSolve;
import edu.stanford.nlp.ling.tokensregex.CoreMapNodePattern.NilAnnotationPattern;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.Index;

public class InferenceWrappers {
	
	public Set<Integer> [] generateZUpdateILP(List<Counter<Integer>> scores, 
											  int numOfMentions, 
											  Set<Integer> goldPos,
											  int nilIndex){
//		System.out.println("Calling ILP inference for Pr (Z | Y,X)");
//		System.out.println("Num of mentions : " + numOfMentions);
//		System.out.println("Relation labels : " + goldPos);

		Set<Integer> [] zUpdate = ErasureUtils.uncheckedCast(new Set[numOfMentions]);
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			zUpdate[mentionIdx] = new HashSet<Integer>(); 
		}
		
		SolverFactory factory = new SolverFactoryLpSolve();
		factory.setParameter(Solver.VERBOSE, 0);
		factory.setParameter(Solver.TIMEOUT, 100); // set timeout to 100 seconds

		Problem problem = new Problem();
		Linear objective = new Linear();
		Linear constraint;

		if(goldPos.size() > numOfMentions){
			//////////////Objective --------------------------------------
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				Counter<Integer> score = scores.get(mentionIdx);
				for(int label : score.keySet()){
					if(label == nilIndex)
						continue; 
					
					String var = "z"+mentionIdx+"_"+"y"+label;
					double coeff = score.getCount(label);
					objective.add(coeff, var);

					//System.out.print(score.getCount(label) + "  " + "z"+mentionIdx+"_"+"y"+label + " + ");
				}
			}
		
			problem.setObjective(objective, OptType.MAX);
			
			/////////// Constraints ------------------------------------------
			
			/// 1. \Sum_{i \in Y'} z_ji = 1 \forall j
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				constraint = new Linear();
				for(int y : goldPos){
					String var = "z"+mentionIdx+"_"+"y"+y;
					constraint.add(1, var);
						
					//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");				
				}
								
				problem.add(constraint, "=", 1);
				//System.out.println(" 0 = "+ "1");
			}
			
			/// 2. \Sum_j z_ji <= 1 \forall i
			for(int y : goldPos){
				constraint = new Linear();
				for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
					String var = "z"+mentionIdx+"_"+"y"+y;
					constraint.add(1, var);
				}
				problem.add(constraint, "<=", 1);
			}
		}	
		else {
			//////////////Objective --------------------------------------
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				Counter<Integer> score = scores.get(mentionIdx);
				for(int label : score.keySet()){
					String var = "z"+mentionIdx+"_"+"y"+label;
					double coeff = score.getCount(label);
					objective.add(coeff, var);

					//System.out.print(score.getCount(label) + "  " + "z"+mentionIdx+"_"+"y"+label + " + ");
				}
			}

			problem.setObjective(objective, OptType.MAX);

			/// 1. equality constraints \Sum_{i \in Y'} z_ji = 1 \forall j
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				constraint = new Linear();
				if(goldPos.size() == 0) { // if goldPos is [] ==> -nil- index
					String var = "z"+mentionIdx+"_"+"y"+nilIndex;
					constraint.add(1, var);
					
					//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");
				}
				else {
					for(int y : goldPos){
						String var = "z"+mentionIdx+"_"+"y"+y;
						constraint.add(1, var);
						
						//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");				
					}
				}
				
				problem.add(constraint, "=", 1);
				//System.out.println(" 0 = "+ "1");
			}

		}
				
		
		
		//System.out.println("\n-----------------");
		/// 2. inequality constraint ===>  1 <= \Sum_j z_ji \forall i \in Y'  {lhs=1, since we consider only Y' i.e goldPos}
		/////////// ------------------------------------------------------
		if(goldPos.size() == 0){ // if goldPos is [] ==> -nil- index
			constraint = new Linear();
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				String var = "z"+mentionIdx+"_"+"y"+nilIndex;
				constraint.add(1, var);
				//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");
			}
			problem.add(constraint, ">=", 1);
			//System.out.println(" 0 - " + "y"+y +" >= 0" );
		}
		else {
			for(int y : goldPos){
				constraint = new Linear();
				for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
					String var = "z"+mentionIdx+"_"+"y"+y;
					constraint.add(1, var);
					//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");
				}
				problem.add(constraint, ">=", 1);
				//System.out.println(" 0 - " + "y"+y +" >= 0" );
			}
		}
		/////////// ------------------------------------------------------
		
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
		
		//System.out.println("Result : " + result);
		
		if(result == null){
			System.out.println("Num of variables : " + problem.getVariablesCount());
			System.out.println("Num of Constraints : " + problem.getConstraintsCount());
			System.out.println("Objective Function : ");
			System.out.println(problem.getObjective());
			System.out.println("Constraints : ");
			for(Constraint c : problem.getConstraints())
				System.out.println(c);
			
			System.out.println("Result is NULL ... Error ...");
			
			System.exit(0);

		}
		
		int numOfUpdates = 0;
		for(Object var : problem.getVariables()) {
			if(result.containsVar(var) && (result.get(var).intValue() == 1)){
				String [] split = var.toString().split("_");
				//System.out.println(split[0]);
				int mentionIdx = Integer.parseInt(split[0].toString().substring(1));
				//System.out.println(split[1]);
				int ylabel = Integer.parseInt(split[1].toString().substring(1));
				zUpdate[mentionIdx].add(ylabel);
				numOfUpdates++;
			}			
		}
		
		if (numOfUpdates != numOfMentions)
		{
			System.out.println(result);
			System.out.println("----------ERROR-----------");
			System.out.println("GOLDPOS : " + goldPos);
		}
		
		//assert (numOfUpdates == numOfMentions); 
		
		return zUpdate;
	}
	
	public YZPredicted generateYZPredictedILP(List<Counter<Integer>> scores,
												  int numOfMentions, 
												  Index<String> yLabelIndex, 
												  Counter<Integer> typeBiasScores,
												  int egId,
												  int epoch,
												  int nilIndex){
		
		YZPredicted predictedVals = new YZPredicted(numOfMentions);
		
		Counter<Integer> yPredicted = predictedVals.getYPredicted();
		int [] zPredicted = predictedVals.getZPredicted();
		
		//System.out.println("Calling ILP inference for Pr (Y,Z | X,T)");
		
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
		/**
		 * Commenting to simulate huffmann
		 */
		//System.out.println();
//		for(String yLabel : yLabelIndex){
//			int y = yLabelIndex.indexOf(yLabel);
//			String var = "y"+y;
//			double coeff = typeBiasScores.getCount(y);
//			objective.add(coeff, var);
//			
//			//System.out.print(typeBiasScores.getCount(y)+" y"+y+" + ");
//		}
		
		problem.setObjective(objective, OptType.MAX);
		/////////// -----------------------------------------------------
		 
		//System.out.println("\n-----------------");
		/////////// Constraints ------------------------------------------

		/// 1. equality constraints \Sum_i z_ji = 1 \forall j
		Linear constraint;
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			constraint = new Linear();
			for(String yLabel : yLabelIndex){
				int y = yLabelIndex.indexOf(yLabel);
				String var = "z"+mentionIdx+"_"+"y"+y;
				constraint.add(1, var);
				
				//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");				
			}
			
			problem.add(constraint, "=", 1); // NOTE : To simulate Hoffmann 
			//System.out.println(" 0 = "+ "1");
		}
		
		//System.out.println("\n-----------------");
		
		/// 2. inequality constraint -- 1 ... z_ji <= y_i \forall j,i
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			for(String yLabel : yLabelIndex){
				constraint = new Linear();
				int y = yLabelIndex.indexOf(yLabel);
				String var1 = "z"+mentionIdx+"_"+"y"+y;
				String var2 = "y"+y;
				constraint.add(1, var1);
				constraint.add(-1, var2);
				problem.add(constraint, "<=", 0);
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
		
		//System.out.println("Result : " + result);
		
		if(result == null){
//			System.out.println("Num of variables : " + problem.getVariablesCount());
//			System.out.println("Num of Constraints : " + problem.getConstraintsCount());
//			System.out.println("Objective Function : ");
//			System.out.println(problem.getObjective());
//			System.out.println("Constraints : ");
//			for(Constraint c : problem.getConstraints())
//				System.out.println(c);
			
			System.out.println("Result is NULL ... Error in iter = " + epoch + " Eg Id : " + egId + " ...  Skipping this");
			
			return predictedVals;

		}
		
		for(Object var : problem.getVariables()) {
			if(result.containsVar(var) && (result.get(var).intValue() == 1)){
				if(var.toString().startsWith("y")) {
					//System.out.println(var + " = " + result.get(var) + " : Y-vars");
					int y = Integer.parseInt(var.toString().substring(1));
					if(y != nilIndex)
						yPredicted.setCount(y, result.get(var).doubleValue());
				}
				else if(var.toString().startsWith("z")) { 
					String [] split = var.toString().split("_");
					//System.out.println(split[0]);
					int mentionIdx = Integer.parseInt(split[0].toString().substring(1));
					//System.out.println(split[1]);
					int ylabel = Integer.parseInt(split[1].toString().substring(1));
					zPredicted[mentionIdx] = ylabel;
					
					//System.out.println(var + " = " + result.get(var) + " : Z-vars");
					
				}
			}
		}
		
		//System.out.println(yPredicted);
		//System.exit(0);
		
		return predictedVals;
	}

}
