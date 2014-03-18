package ark.data.annotation;

import java.util.HashMap;
import java.util.Map;

import ark.data.annotation.nlp.TokenSpan;

public abstract class Datum<L> {	
	protected int id;
	protected L label;
	
	public int getId() {
		return this.id;
	}
	
	public L getLabel() {
		return this.label;
	}
	
	@Override
	public int hashCode() {
		// FIXME: Make better
		return this.id;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public boolean equals(Object o) {
		Datum<L> datum = (Datum<L>)o;
		return datum.id == this.id;
	}
	
	public interface StringExtractor<D extends Datum<L>, L> {
		String toString();
		String[] extract(D datum);
	}
	
	public interface TokenSpanExtractor<D extends Datum<L>, L> {
		String toString();
		TokenSpan[] extract(D datum);
	}
	
	public abstract class AnnotationTools<D extends Datum<L>> {
		protected Map<String, Datum.TokenSpanExtractor<D, L>> tokenSpanExtractors;
		protected Map<String, Datum.StringExtractor<D, L>> stringExtractors;

		public AnnotationTools() {
			this.tokenSpanExtractors = new HashMap<String, Datum.TokenSpanExtractor<D, L>>();
			this.stringExtractors = new HashMap<String, Datum.StringExtractor<D, L>>();
		}
		
		public Datum.TokenSpanExtractor<D, L> getTokenSpanExtractor(String name) {
			return this.tokenSpanExtractors.get(name);
		}
		
		public Datum.StringExtractor<D, L> getStringExtractor(String name) {
			return this.stringExtractors.get(name);
		}
		
		public boolean addTokenSpanExtractor(Datum.TokenSpanExtractor<D, L> tokenSpanExtractor) {
			this.tokenSpanExtractors.put(tokenSpanExtractor.toString(), tokenSpanExtractor);
			return true;
		}
		
		public boolean addStringExtractor(Datum.StringExtractor<D, L> stringExtractor) {
			this.stringExtractors.put(stringExtractor.toString(), stringExtractor);
			return true;
		}
		
		public abstract L labelFromString(String str);
	}
}
