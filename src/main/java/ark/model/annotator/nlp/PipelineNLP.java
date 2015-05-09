package ark.model.annotator.nlp;

import java.util.List;
import java.util.Map;
import ark.data.annotation.nlp.AnnotationTypeNLP;
import ark.data.annotation.nlp.DocumentNLP;
import ark.data.annotation.nlp.TokenSpan;
import ark.model.annotator.Pipeline;
import ark.util.Pair;

public class PipelineNLP extends Pipeline {
	protected DocumentNLP document;
	
	public PipelineNLP() {
		super();
	}
	
	public boolean setDocument(DocumentNLP document) {
		this.document = document;
		return true;
	}
	
	@SuppressWarnings("unchecked")
	public <T> T annotateDocument(AnnotationTypeNLP<T> annotationType) {
		AnnotatorDocument<T> annotator = (AnnotatorDocument<T>)this.annotators.get(annotationType);
		return annotator.annotate(this.document);
	}
	
	@SuppressWarnings("unchecked")
	public <T> Map<Integer, T> annotateSentences(AnnotationTypeNLP<T> annotationType) {
		AnnotatorSentence<T> annotator = (AnnotatorSentence<T>)this.annotators.get(annotationType);
		return annotator.annotate(this.document);
	}
	
	@SuppressWarnings("unchecked")
	public <T> List<Pair<TokenSpan, T>> annotateTokenSpans(AnnotationTypeNLP<T> annotationType) {
		AnnotatorTokenSpan<T> annotator = (AnnotatorTokenSpan<T>)this.annotators.get(annotationType);
		return annotator.annotate(this.document);
	}
	
	@SuppressWarnings("unchecked")
	public <T> T[][] annotateTokens(AnnotationTypeNLP<T> annotationType) {
		AnnotatorToken<T> annotator = (AnnotatorToken<T>)this.annotators.get(annotationType);
		return annotator.annotate(this.document);
	}
	
	public PipelineNLP weld(PipelineNLP pipeline) {
		PipelineNLP welded = new PipelineNLP();
		welded.annotators.putAll(this.annotators);
		welded.annotators.putAll(pipeline.annotators);
		return welded;
	}
}
