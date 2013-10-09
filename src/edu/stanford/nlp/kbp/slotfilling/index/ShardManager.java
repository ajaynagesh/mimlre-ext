package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import edu.stanford.nlp.kbp.slotfilling.common.ProcessWrapper;

/**
 * Runs and manages multiple LucenePipelineCacher processes.
 * Each job is run in a different process and is managed by a separate thread in this JVM
 * This is useful to parse multiple index shards simultaneously on multi-core machine, e.g., jackfruit 
 */
public class ShardManager {
  private static final String JOB_COMMAND = "/juicy/scr61/scr/nlp/mihais/Wikipedia_shards/nohup_one_shard.sh";
  private static final String LOG_DIR = "/juicy/scr61/scr/nlp/mihais/Wikipedia_shards/completed_logs";
  private static final int THREAD_COUNT = 6;
  
  static List<Integer> SHARDS = new ArrayList<Integer>();
  List<Integer> completedShards = new ArrayList<Integer>();
  
  private static void loadShards(String fn) throws Exception {
    BufferedReader is = new BufferedReader(new FileReader(fn));
    for(String line; (line = is.readLine()) != null; ){
      SHARDS.add(Integer.valueOf(line.trim()));
    }
    is.close();
    System.err.println("Loaded " + SHARDS.size() + " shards.");
  }
  
  public synchronized void run() {
    // start threads
    List<Runnable> jobs = createJobs();  
    ExecutorService threadPool = Executors.newFixedThreadPool(THREAD_COUNT);
    for(Runnable job: jobs){
      threadPool.execute(job);
    }
    
    // wait for all threads to complete
    this.waitForThreads();
    threadPool.shutdown();
  }
  
  private List<Runnable> createJobs() {
    List<Runnable> jobs = new ArrayList<Runnable>();
    for(int shard: SHARDS) jobs.add(new ShardManagerThread(this, shard));
    return jobs;
  }
  
  static class ShardManagerThread implements Runnable {
    private ShardManager manager;
    private int shard;
    
    public ShardManagerThread(ShardManager m, int s) {
      this.manager = m;
      this.shard = s;
    }
    
    @Override
    public void run() {
      String cmd = JOB_COMMAND + " " + shard;
      
      int exitCode = -1;
      try {
        PrintStream log = new PrintStream(new FileOutputStream(LOG_DIR + File.separator + shard + ".log"));
        exitCode = launch(cmd, log);
        log.close();
      } catch(Exception e) {
        System.err.println("ERROR: Command failed: " + cmd);
      }
      
      manager.reportCompletion(shard, exitCode);
      manager.threadFinished(shard);
    }    
  }
  
  private static int launch(String cmd, PrintStream os) throws IOException, InterruptedException {
    System.err.println("Launching command: " + cmd);
    
    ProcessWrapper process = ProcessWrapper.create(cmd);
    process.waitFor();
    int exitCode = process.exitValue();
    
    String err = process.consumeErrorStream();
    if(err.length() > 0){
      os.println("Error stream contained:\n" + err);
    }
    String stdout = process.consumeReadStream();
    if(stdout.length() > 0){
      os.println("Stdout stream contained:\n" + stdout);
    }
    
    return exitCode;
 }
  
  private synchronized void waitForThreads() {
    while(completedShards.size() < SHARDS.size()){
      try {
        this.wait();
      } catch(InterruptedException e) {
        System.err.println("Main thread interrupted!\n");
        break;
      }
    }
    System.err.println("All threads finished.\n");
  }
  
  public synchronized void threadFinished(int shard) {
    completedShards.add(shard);
    this.notify();
  }
  
  public synchronized void reportCompletion(int shard, int exitCode) {
    System.err.println("Shard #" + shard + " completed with exit code " + exitCode +
        (exitCode == 0 ? " (SUCCESS)" : " (ERROR)"));
  }
  
  public static void main(String[] args) throws Exception {
    loadShards(args[0]);
    ShardManager sm = new ShardManager();
    sm.run();
  }
}
