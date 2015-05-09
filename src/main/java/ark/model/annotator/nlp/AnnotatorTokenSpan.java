package ark.model.annotator.nlp;

import java.util.List;

import ark.data.annotation.nlp.DocumentNLP;
import ark.data.annotation.nlp.TokenSpan;
import ark.model.annotator.Annotator;
import ark.util.Pair;

public interface AnnotatorTokenSpan<T> extends Annotator<T> {
	List<Pair<TokenSpan, T>> annotate(DocumentNLP document);
}
