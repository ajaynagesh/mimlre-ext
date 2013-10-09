
package edu.stanford.nlp.kbp.slotfilling.common;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;

/**
 * Wrapper for a Process object
 * @author Mihai
 */
public class ProcessWrapper {
	/** The actual process object */
	private Process mProcess;

	/** The command that created this process */
	private String mCommand;

	/** The error stream */
	private BufferedReader mErrorStream;

	/** The read stream */
	private BufferedReader mReadStream;

	/** The write stream */
	private PrintStream mWriteStream;

	private ProcessWrapper(String cmd) { mCommand = cmd; }

	public static ProcessWrapper create(String cmd) 
	throws java.io.IOException {

		ProcessWrapper pw = new ProcessWrapper(cmd);
		pw.mProcess = Runtime.getRuntime().exec(cmd);
		pw.mErrorStream = 
			new BufferedReader(new InputStreamReader(pw.mProcess.getErrorStream()));
		pw.mReadStream = 
			new BufferedReader(new InputStreamReader(pw.mProcess.getInputStream()));
		pw.mWriteStream = 
			new PrintStream(pw.mProcess.getOutputStream());
		return pw;
	}

	/**
	 * Close all stream and stop the process
	 */
	public void stop() throws java.io.IOException {
		mWriteStream.close();
		mErrorStream.close();
		mReadStream.close();
		mProcess.destroy();
	}

	/**
	 * Consume any bytes available in the error stream
	 */
	public String consumeErrorStream() throws java.io.IOException {
		StringBuffer buffer = new StringBuffer();
		while(mErrorStream.ready()){
			char c = (char) mErrorStream.read();
			buffer.append(c);
		}
		return buffer.toString().trim();
	}
	
	/**
   * Consume any bytes available at stdout
   */
  public String consumeReadStream() throws java.io.IOException {
    StringBuffer buffer = new StringBuffer();
    while(mReadStream.ready()){
      char c = (char) mReadStream.read();
      buffer.append(c);
    }
    return buffer.toString().trim();
  }

	public String getCommand() { return mCommand; }

	public PrintStream getWriteStream() { return mWriteStream; }
	public BufferedReader getReadStream() { return mReadStream; }

	public boolean checkError() { return mWriteStream.checkError(); }
	public void flush() { mWriteStream.flush(); }
	public void print(boolean b) { mWriteStream.print(b); }
	public void print(char c) { mWriteStream.print(c); }
	public void print(char[] s) { mWriteStream.print(s); }
	public void print(double d) { mWriteStream.print(d); }
	public void print(float f) { mWriteStream.print(f); }
	public void print(int i) { mWriteStream.print(i); }
	public void print(long l) { mWriteStream.print(l); }
	public void print(Object obj) { mWriteStream.print(obj); }
	public void print(String s) { mWriteStream.print(s); }
	public void println() { mWriteStream.println(); flush(); }
	public void println(boolean x) { mWriteStream.println(x); flush(); }
	public void println(char x) { mWriteStream.println(x); flush(); }
	public void println(char[] x) { mWriteStream.println(x); flush(); }
	public void println(double x) { mWriteStream.println(x); flush(); }
	public void println(float x) { mWriteStream.println(x); flush(); }
	public void println(int x) { mWriteStream.println(x); flush(); }
	public void println(long x) { mWriteStream.println(x); flush(); }
	public void println(Object x) { mWriteStream.println(x); flush(); }
	public void println(String x) { mWriteStream.println(x); flush(); }

	public boolean ready() throws java.io.IOException { 
		return mReadStream.ready(); 
	}
	public int read() throws java.io.IOException { 
		return mReadStream.read(); 
	}
	public String readLine() throws java.io.IOException { 
		return mReadStream.readLine(); 
	}

	public int waitFor() throws InterruptedException {
		return mProcess.waitFor();
	}

	public int exitValue() {
		return mProcess.exitValue();
	}
}
