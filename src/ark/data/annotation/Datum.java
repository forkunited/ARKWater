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
	
	public static abstract class Tools<D extends Datum<L>, L> {
		public static interface StringExtractor<D extends Datum<L>, L> {
			String toString();
			String[] extract(D datum);
		}
		
		public static interface TokenSpanExtractor<D extends Datum<L>, L> {
			String toString();
			TokenSpan[] extract(D datum);
		}
		
		protected Map<String, TokenSpanExtractor<D, L>> tokenSpanExtractors;
		protected Map<String, StringExtractor<D, L>> stringExtractors;

		public Tools() {
			this.tokenSpanExtractors = new HashMap<String, TokenSpanExtractor<D, L>>();
			this.stringExtractors = new HashMap<String, StringExtractor<D, L>>();
		}
		
		public TokenSpanExtractor<D, L> getTokenSpanExtractor(String name) {
			return this.tokenSpanExtractors.get(name);
		}
		
		public StringExtractor<D, L> getStringExtractor(String name) {
			return this.stringExtractors.get(name);
		}
		
		public boolean addTokenSpanExtractor(TokenSpanExtractor<D, L> tokenSpanExtractor) {
			this.tokenSpanExtractors.put(tokenSpanExtractor.toString(), tokenSpanExtractor);
			return true;
		}
		
		public boolean addStringExtractor(StringExtractor<D, L> stringExtractor) {
			this.stringExtractors.put(stringExtractor.toString(), stringExtractor);
			return true;
		}
		
		public abstract L labelFromString(String str);
	}
}
