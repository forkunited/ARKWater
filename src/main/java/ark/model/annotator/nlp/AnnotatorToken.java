package ark.model.annotator.nlp;

import ark.data.annotation.nlp.DocumentNLP;
import ark.model.annotator.Annotator;

public interface AnnotatorToken<T> extends Annotator<T> {
	T[][] annotate(DocumentNLP document);
}