package edu.stanford.nlp.topicmodeling.topicflow.graphviz_layout;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.Writer;
import java.util.Iterator;

public class FileUtils
{
//------------------------------------------------------------------------------
	public static final void makeDirectory( String path )
	{
		final File folder = new File( path );
		folder.mkdirs();
	}
//------------------------------------------------------------------------------	
	public static final int BUFFER_SIZE = 65536;
	public static final void writeToFile( String filename, String text )
	{
		writeToFile( new File( filename ), text );
	}
	public static final void writeToFile( File file, String text )
	{
		try
		{
			final Writer writer = new FileWriter( file );
			final BufferedWriter bufferedWriter = new BufferedWriter( writer );
			
			bufferedWriter.write( text, 0, text.length() );
			
			bufferedWriter.flush();
			bufferedWriter.close();
			writer.close();
		}
		catch ( IOException e )
		{
			e.printStackTrace();
		}
	}
	public static final String readFromFile( String filename )
	{
		return readFromFile( new File( filename ) );
	}
	public static final String readFromFile( File file )
	{
		final StringBuffer s = new StringBuffer();
		try
		{
			final Reader reader = new FileReader( file );
			final BufferedReader bufferedReader = new BufferedReader( reader );
			
			final char[] chars = new char[ BUFFER_SIZE ];
			int size = 0;
			while ( ( size = bufferedReader.read( chars ) ) > 0 )
				s.append( String.valueOf( chars, 0, size ) );

			bufferedReader.close();
			reader.close();
		}
		catch ( IOException e )
		{
			e.printStackTrace();
		}
		return s.toString();
	}
//------------------------------------------------------------------------------

	public static final Iterable<String> readLines( String filename )
	{
		return readLines( new File( filename ) );
	}
	public static final Iterable<String> readLines( File file ) 
	{
		try
		{
			final Reader reader = new FileReader( file );
			final BufferedReader bufferedReader = new BufferedReader( reader );
			return new LineReaderIterable( bufferedReader );
		}
		catch ( FileNotFoundException e )
		{
			e.printStackTrace();
		}
		return null;
	}
	public static final PrintWriter writeLines( String filename )
	{
		return writeLines( new File( filename ) );
	}
	public static final PrintWriter writeLines( File file )
	{
		try
		{
			final Writer writer = new FileWriter( file );
			final BufferedWriter bufferedWriter = new BufferedWriter( writer );
			return new PrintWriter( bufferedWriter );
		}
		catch ( FileNotFoundException e )
		{
			e.printStackTrace();
		}
		catch ( IOException e ) 
		{
			e.printStackTrace();
		}
		return null;
	}
	public static final class LineReaderIterable implements Iterable<String>
	{
		private final BufferedReader reader;
		private LineReaderIterable( BufferedReader reader )
		{
			this.reader = reader;
		}
		@Override
		public Iterator<String> iterator()
		{
			return new LineIterator();
		}
		public final class LineIterator implements Iterator<String>
		{
			private LineIterator() { }
			@Override
			public boolean hasNext()
			{
				try 
				{
					if ( reader.ready() )
						return true;
					else
					{
						reader.close();
						return false;
					}
				}
				catch (IOException e) 
				{
					e.printStackTrace();
				}
				return false;
			}
			@Override
			public String next()
			{
				try 
				{
					return reader.readLine();
				}
				catch (IOException e) 
				{
					e.printStackTrace();
				}
				return null;
			}
			@Override
			public void remove()
			{
				throw new UnsupportedOperationException();
			}
		}
	}
}
