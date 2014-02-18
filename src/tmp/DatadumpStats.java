package tmp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class DatadumpStats {
	
	public static void readFile(String stats_string) throws IOException{
		
		BufferedReader br = new BufferedReader(new FileReader(new File(stats_string))); 
		String line;
		int numoflines = 0;
		HashMap<String, HashMap<String, Integer>> dataStats = new HashMap<String, HashMap<String,Integer>>();

		while((line = br.readLine()) != null){
		
			String [] info = line.split("\\/");
			
			String relation = info[2];
			String arg1_arg2 = info[0]+"_"+info[1];
			
			if(! dataStats.containsKey(relation)){
				HashMap<String, Integer> h_argtypeStats = new HashMap<String, Integer>();
				h_argtypeStats.put(arg1_arg2, 1);
				dataStats.put(relation, h_argtypeStats);
			}
			else {
				HashMap<String, Integer> h_argtypeStats = dataStats.get(relation);
				
				if(! h_argtypeStats.containsKey(arg1_arg2)){
					h_argtypeStats.put(arg1_arg2, 1);
				}
				else {
					int cnt = h_argtypeStats.get(arg1_arg2);
					cnt++;
					h_argtypeStats.put(arg1_arg2, cnt);
				}
			}
			
			
			numoflines++;
		}
		HashSet<String> arg_types = new HashSet<String>();
		for(String relation : dataStats.keySet()){
			//System.out.print(relation + " ["); 
			HashMap<String, Integer> rel_types = dataStats.get(relation);
			for(String argtype : rel_types.keySet()){
				arg_types.add(argtype);
				//System.out.print(argtype+"="+rel_types.get(argtype)+", ");
			}
			//System.out.println("]");
		}
		//System.out.println("-------------");

		ArrayList<String> argTypesArray = new ArrayList<String>(arg_types);
		
		System.out.print("\t");
		for(String s : argTypesArray){
			System.out.print(s+"\t");
		}
		System.out.println();
			
			
		for(String relation : dataStats.keySet()){
			HashMap<String, Integer> rel_types = dataStats.get(relation);
			System.out.print(relation + "\t");
			for(String s : argTypesArray){
				if(!rel_types.containsKey(s))
					System.out.print(0 + "\t");
				else{
					System.out.print(rel_types.get(s) + "\t");
				}
			}
			System.out.println();
		}
		
		System.out.println(argTypesArray.size());
		
	}
	
	public static void main(String args[]) throws IOException{
		String stats_file = "/home/ajay/Desktop/datadump.txt";
		readFile(stats_file);
	}
	
}
