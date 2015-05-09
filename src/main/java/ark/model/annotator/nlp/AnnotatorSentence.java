package ark.model.annotator.nlp;

import java.util.Map;

import ark.data.annotation.nlp.DocumentNLP;
import ark.model.annotator.Annotator;

public interface AnnotatorSentence<T> extends Annotator<T> {
	Map<Integer, T> annotate(DocumentNLP document);
}
