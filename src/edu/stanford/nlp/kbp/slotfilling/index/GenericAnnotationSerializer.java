package edu.stanford.nlp.kbp.slotfilling.index;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;

import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationSerializer;

/**
 * Serializes Annotation objects using the default Java serializer
 */
public class GenericAnnotationSerializer implements AnnotationSerializer {

  @Override
  public Annotation load(InputStream is) throws IOException, ClassNotFoundException, ClassCastException {
    ObjectInputStream objectInput = new ObjectInputStream(is);
    Object annotation = objectInput.readObject();
    if(annotation == null) return null;
    if(! (annotation instanceof Annotation)){
      throw new ClassCastException("ERROR: Serialized data does not contain an Annotation!");
    }
    return (Annotation) annotation;
  }

  @Override
  public void save(Annotation corpus, OutputStream os) throws IOException {
    ObjectOutputStream objectOutput = new ObjectOutputStream(os);
    objectOutput.writeObject(corpus);
    objectOutput.close();
  }
  
}
