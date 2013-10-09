package edu.stanford.nlp.kbp.slotfilling.index;

import edu.stanford.nlp.util.XMLUtils;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Timing;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.WordToSentenceProcessor;
import edu.stanford.nlp.ling.CoreLabel;

import java.io.*;
import java.util.*;
import java.util.regex.*;
import java.util.zip.*;
import java.util.concurrent.*;

public class WikipediaReader {

	private static final Pattern startPattern = Pattern.compile("<text.*?>", Pattern.CASE_INSENSITIVE);
	private static final Pattern endPattern = Pattern.compile("</text>", Pattern.CASE_INSENSITIVE);

	private static final Pattern header = Pattern.compile("(?:^|\n)=+\\s*(.*?)\\s*(=+)(?:$|\n)");
	private static final Pattern articleEndPattern = Pattern.compile("(Citations)|(See also)|(External links)|(General resources)|(Bibliography)|(Biographical and bibliographical)|(External references)|(References)", Pattern.CASE_INSENSITIVE);


	private WikipediaReader() {
	}


	private static boolean doArticle(BufferedReader in, PrintWriter out, int start, int end) throws IOException {

		if (articleID >= end) { return false; }

		String article;
		while ((article = readArticle(in)) == null || article.length() == 0) {
			if (article == null) { return false; }
		}

		if (articleID >= start && articleID < end) {
			System.err.print("["+articleID+"]");

			final StringBuilder processedArticle = new StringBuilder();
			final Semaphore semaphore = new Semaphore(1);
			final String articleFinal = article;
			Timing timer = new Timing();

			try { semaphore.acquire(); } catch (Exception e) {
				System.err.println("Couldn't get a semaphore!");
			}
			timer.start();

			new Thread() {
				public void run() {
					processArticle(articleFinal, processedArticle);
					semaphore.release();
				}
			}.start();

			while (semaphore.availablePermits() == 0 && timer.report() < 10000) {
				try {
					Thread.sleep(1);
				} catch (Exception e) {
					// ignore
				}
			}

			if (semaphore.availablePermits() == 1) {
				out.println("<article id=\""+articleID+"\">");
				out.println(processedArticle.toString());
				out.println("</article>");
			}
		}

		articleID++;

		return true;
	}


	private static String readArticle(BufferedReader in) throws IOException {

		String line;
		StringBuilder article = new StringBuilder();

		while ((line = in.readLine()) != null) {
			Matcher m = startPattern.matcher(line);
			if (m.find()) {
				line = line.substring(m.end());
				if (line.length() >= 9 && line.substring(0,9).equalsIgnoreCase("#REDIRECT")) { continue; }
				break;
			}
		}

		if (line == null) { return null; }

		boolean skip = false;

		while (line != null) {
			Matcher m = endPattern.matcher(line);
			if (m.find()) {
				line = line.substring(0, m.start());
				line = fixLine(line);
				m = header.matcher(line);
				if (m.matches() && articleEndPattern.matcher(m.group(1)).matches()) {
					skip = true;
				}
				if (!skip) {
					article.append(line).append("\n");
				}
				break;
			}
			line = fixLine(line);
			m = header.matcher(line);
			if (m.matches() && articleEndPattern.matcher(m.group(1)).matches()) {
				skip = true;
			}
			if (!skip) {
				article.append(line).append("\n");
			}
			line = in.readLine();
		}

		return article.toString().trim();
	}

	private static String fixLine(String orig) {
		return orig.trim();
	}


	private static final String notCurlyBrace = "[^\\{\\}]";
	private static final Pattern bracket0 = Pattern.compile("\\{"+notCurlyBrace+"*\\}", Pattern.DOTALL);

	private static final String notSquareBracketOrColon = "[^\\[\\]\\:]";
	private static final String notSquareBracket = "[^\\[\\]]";
	private static final String notSquareBracketOrSpace = "[^\\[\\] ]";
	// This RE saves the section in between the brackets which normally
	// gets displayed to the user.  It skips the text before the first |.
	// For example:
	// [[foo]] -> foo
	// [[foo|bar]] -> bar
	// [[foo|bar|baz]] -> bar|baz
	private static final Pattern bracket1 = Pattern.compile("\\[\\[(?:"+notSquareBracketOrColon+"*?\\|)?("+notSquareBracketOrColon+"*?)\\]\\]", Pattern.DOTALL);

	private static final Pattern bracket2 = Pattern.compile("\\[\\[" + notSquareBracket + "*?:" + notSquareBracket + "*?\\]\\]", Pattern.DOTALL);

	private static final Pattern bracket3 = Pattern.compile("\\[" + notSquareBracketOrSpace + "* (" + notSquareBracket + "*)\\]", Pattern.DOTALL);

	private static final Pattern bracket4 = Pattern.compile("\\[" + notSquareBracketOrSpace + "*\\]", Pattern.DOTALL);

	private static final Pattern ref1 = Pattern.compile("<ref[^>]*?/>", Pattern.CASE_INSENSITIVE+Pattern.DOTALL);
	private static final Pattern ref2 = Pattern.compile("<ref[^>]*?>.*?</ref>", Pattern.CASE_INSENSITIVE+Pattern.DOTALL);
	private static final Pattern sgml = Pattern.compile("<.*?>", Pattern.DOTALL);
	private static final Pattern comment = Pattern.compile("<\\!--.*?-->", Pattern.DOTALL);
	private static final Pattern newLine = Pattern.compile("\n+", Pattern.DOTALL);
	private static final Pattern space = Pattern.compile("&nbsp;");
	private static final Pattern multipleSpaces = Pattern.compile(" +", Pattern.DOTALL);

	private static final Pattern smallSubsection = Pattern.compile("====([^=]+)====");
	private static final Pattern subsection = Pattern.compile("===([^=]+)===");
	private static final Pattern section = Pattern.compile("==([^=]+)==");

	public enum ListRemoval {
		IGNORE, REMOVE, COMMA;
	}
	// This RE detects the *first* list element in a list of things;
	// this is useful for if you want to do one thing with the first
	// element and something else with the other elements
	private static final Pattern listFirstLine = Pattern.compile("((?:\\A|\n)[ \t]*([^ *#\t\n][^\n]*)?)\n[ \t]*[*#]+");
	// This detects a list at the very start of a file without any other
	// content between the start and the list.  It can't be combined
	// with the "lists" pattern because we want to do two different
	// things to those lists.
	private static final Pattern listAtBeginning = Pattern.compile("\\A[\\t ]*[*#]+");
	private static final Pattern lists = Pattern.compile("\n[\\t ]*[*#]+");

	private static int articleID = 0;



	public static String removeMarkup(String orig, boolean escapedXML,
			ListRemoval listRemoval) {
		String line = orig;

		Matcher m;

		// remove all the section headings
		// TODO: do we want to make keeping them an option?
		m = smallSubsection.matcher(line);
		line = m.replaceAll("");

		m = subsection.matcher(line);
		line = m.replaceAll("");

		m = section.matcher(line);
		line = m.replaceAll("");

		m = bracket0.matcher(line);
		while (m.find()) {
			line = m.replaceAll("");
			m = bracket0.matcher(line);
		}

		m = bracket1.matcher(line);
		while (m.find()) {
			line = m.replaceAll("$1");
			m = bracket1.matcher(line);
		}

		m = bracket2.matcher(line);
		line = m.replaceAll("");

		m = bracket3.matcher(line);
		line = m.replaceAll("$1");

		m = bracket4.matcher(line);
		line = m.replaceAll("");

		if (escapedXML)
			line = XMLUtils.unescapeStringForXML(line);

		m = comment.matcher(line);
		line = m.replaceAll("");

		m = ref1.matcher(line);
		line = m.replaceAll("");

		m = ref2.matcher(line);
		line = m.replaceAll("");

		m = sgml.matcher(line);
		line = m.replaceAll("");

		if (escapedXML)
			line = XMLUtils.escapeXML(line);

		switch (listRemoval) {
		case IGNORE: break;
		case REMOVE:
			m = listAtBeginning.matcher(line);
			line = m.replaceAll("");
			m = lists.matcher(line);
			line = m.replaceAll("");
			break;
		case COMMA:
			m = listFirstLine.matcher(line);
			line = m.replaceAll("$1");
			m = listAtBeginning.matcher(line);
			line = m.replaceAll("");
			m = lists.matcher(line);
			line = m.replaceAll(",");
			break;
		default:
			throw new AssertionError("Unknown list removal option specified " +
					listRemoval);
		}

		m = header.matcher(line);
		while (m.find()) {
			line = m.replaceAll("\n<header>$1</header>\n");
			m = header.matcher(line);
		}

		m = newLine.matcher(line);
		line = m.replaceAll("\n");

		m = space.matcher(line);
		line = m.replaceAll(" ");

		m = multipleSpaces.matcher(line);
		line = m.replaceAll(" ");
		return line.trim();
	}

	private static void processArticle(String orig, StringBuilder processed) {
		String line = removeMarkup(orig, true, ListRemoval.IGNORE);

		PTBTokenizer ptb = PTBTokenizer.newPTBTokenizer(new StringReader(line), false, true);
		WordToSentenceProcessor<CoreLabel> wts = new WordToSentenceProcessor<CoreLabel>();

		CoreLabel token = null;
		List<List<CoreLabel>> sections = new ArrayList<List<CoreLabel>>();
		List<CoreLabel> section = new ArrayList<CoreLabel>();
		boolean prevWordIsPunct = false;

		while (ptb.hasNext()) {
			token = (CoreLabel)ptb.next();

			if (token.originalText().equals("<header>")) {
				if (!section.isEmpty()) { sections.add(section); }
				section = new ArrayList<CoreLabel>();
				section.add(token);
				prevWordIsPunct = false;
			}  else if (token.originalText().equals("</header>")) {
				section.add(token);
				sections.add(section);
				section = new ArrayList<CoreLabel>();
				prevWordIsPunct = false;
			} else {
				String word = XMLUtils.unescapeStringForXML(token.originalText());
				if (prevWordIsPunct && word.length() == 1 && StringUtils.isPunct(word)) {
					int index = section.size()-1;
					CoreLabel prevToken = section.get(index);
					prevToken.setOriginalText(prevToken.originalText()+prevToken.after()+token.originalText());
					prevToken.setAfter(token.after());
				} else {
					section.add(token);
				}
				prevWordIsPunct = StringUtils.isPunct(word);
			}
		}
		if (!section.isEmpty()) { sections.add(section); }

		List<List<CoreLabel>> sentences = new ArrayList<List<CoreLabel>>();
		for (List<CoreLabel> sec : sections) {
			if (sec.get(0).originalText().equals("<header>")) {
				sentences.add(sec);
			} else {
				sentences.addAll(wts.process(sec));
			}
		}

		int sentenceID = -1;
		for (List<CoreLabel> sentence : sentences) {
			boolean first = true;
			boolean header = sentence.get(0).originalText().equals("<header>");
			int wordID = 0;
			for (CoreLabel cl : sentence) {
				token = cl;
				processed.append(token.before());

				if (first) {
					sentenceID++;
					processed.append("<sentence id=\"").append(articleID).append(".").append(sentenceID).append("\" header=\"").append(header).append("\">");
					first = false;
					if (header) { continue; }
				}
				if (cl.originalText().equals("</header>")) { continue; }

				Matcher m = sgml.matcher(token.originalText());
				if (m.matches()) {
					processed.append(token.originalText());
				} else {
					processed.append("<w id=\"").append(articleID).append(".").append(sentenceID).append(".").append(wordID).append("\">");
					processed.append(token.originalText());
					processed.append("</w>");
					wordID++;
				}
			}
			processed.append("</sentence>");

		}
		if (token != null) {
			processed.append(token.after());
		}
	}

	public static void main(String[] args) throws IOException {
		BufferedReader in = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(args[0]))));
		PrintWriter out = new PrintWriter(new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(args[1]))));
		int start = Integer.parseInt(args[2]);
		int end = Integer.parseInt(args[3]);

		out.println("<wikipedia>");
		while (doArticle(in, out, start, end)) {}
		out.println("</wikipedia>");

		out.flush();
		out.close();

	}

}