package edu.stanford.nlp.kbp.slotfilling.common;

import java.util.logging.ConsoleHandler;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class Log {
  public static final Logger logger;


  private Log() {} // static class


  static {
    logger = Logger.getLogger(Log.class.getName());
  }

  public static void setLevel(Level level) {
    setConsoleLevel(Level.FINEST);
    logger.setLevel(level);
  }

  public static Level stringToLevel(String s) {
    if(s.equalsIgnoreCase("finest")) return Level.FINEST;
    if(s.equalsIgnoreCase("finer")) return Level.FINER;
    if(s.equalsIgnoreCase("fine")) return Level.FINE;
    if(s.equalsIgnoreCase("info")) return Level.INFO;
    if(s.equalsIgnoreCase("severe")) return Level.SEVERE;
    throw new RuntimeException("Unknown log level: " + s);
  }

  public static Level getLevel() {
    return logger.getLevel();
  }

  public static boolean levelFinerThan(Level level) {
    assert(getLevel() != null);
    return getLevel().intValue() <= level.intValue();
  }

  public static void info(String s) { logger.info(s); }
  public static void fine(String s) { logger.fine(s); }
  public static void finest(String s) { logger.finest(s); }
  public static void severe(String s) { logger.severe(s); }

  private static void setConsoleLevel(Level level) {
    // get the top Logger:
    Logger topLogger = java.util.logging.Logger.getLogger("");

    // Handler for console (reuse it if it already exists)
    Handler consoleHandler = null;
    // see if there is already a console handler
    for (Handler handler : topLogger.getHandlers()) {
      if (handler instanceof ConsoleHandler) {
        // found the console handler
        consoleHandler = handler;
        break;
      }
    }

    if (consoleHandler == null) {
      // there was no console handler found, create a new one
      consoleHandler = new ConsoleHandler();
      topLogger.addHandler(consoleHandler);
    }
    // set the console handler level:
    consoleHandler.setLevel(level);
    consoleHandler.setFormatter(new SimpleFormatter());
  }
}
