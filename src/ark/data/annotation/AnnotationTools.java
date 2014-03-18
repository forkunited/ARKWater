package ark.data.annotation;

import java.util.HashMap;
import java.util.Map;

import ark.data.annotation.nlp.TokenSpan;

public abstract class AnnotationTools<D extends Datum<L>, L> {
	public interface StringExtractor<D extends Datum<L>, L> {
		String toString();
		String[] extract(D datum);
	}
	
	public interface TokenSpanExtractor<D extends Datum<L>, L> {
		String toString();
		TokenSpan[] extract(D datum);
	}
	
	protected Map<String, AnnotationTools.TokenSpanExtractor<D, L>> tokenSpanExtractors;
	protected Map<String, AnnotationTools.StringExtractor<D, L>> stringExtractors;

	public AnnotationTools() {
		this.tokenSpanExtractors = new HashMap<String, AnnotationTools.TokenSpanExtractor<D, L>>();
		this.stringExtractors = new HashMap<String, AnnotationTools.StringExtractor<D, L>>();
	}
	
	public AnnotationTools.TokenSpanExtractor<D, L> getTokenSpanExtractor(String name) {
		return this.tokenSpanExtractors.get(name);
	}
	
	public AnnotationTools.StringExtractor<D, L> getStringxtractor(String name) {
		return this.stringExtractors.get(name);
	}
	
	public boolean addTokenSpanExtractor(AnnotationTools.TokenSpanExtractor<D, L> tokenSpanExtractor) {
		this.tokenSpanExtractors.put(tokenSpanExtractor.toString(), tokenSpanExtractor);
		return true;
	}
	
	public boolean addTokenSpanExtractor(AnnotationTools.StringExtractor<D, L> stringExtractor) {
		this.stringExtractors.put(stringExtractor.toString(), stringExtractor);
		return true;
	}
	
	public abstract L labelFromString(String str);
}
