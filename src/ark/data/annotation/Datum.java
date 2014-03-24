package ark.data.annotation;

import java.util.HashMap;
import java.util.Map;

import ark.data.DataTools;
import ark.data.annotation.nlp.TokenSpan;
import ark.data.feature.Feature;
import ark.data.feature.FeatureGazetteerContains;
import ark.data.feature.FeatureGazetteerEditDistance;
import ark.data.feature.FeatureGazetteerInitialism;
import ark.data.feature.FeatureGazetteerPrefixTokens;
import ark.data.feature.FeatureNGramContext;
import ark.data.feature.FeatureNGramDep;
import ark.data.feature.FeatureNGramSentence;
import ark.model.SupervisedModel;
import ark.model.SupervisedModelCreg;

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
		
		public static interface LabelMapping<L> {
			String toString();
			L map(L label);
		}
		
		protected DataTools dataTools;
		
		protected Map<String, TokenSpanExtractor<D, L>> tokenSpanExtractors;
		protected Map<String, StringExtractor<D, L>> stringExtractors;
		protected Map<String, LabelMapping<L>> labelMappings;
		
		protected Map<String, Feature<D, L>> genericFeatures;
		protected Map<String, SupervisedModel<D, L>> genericModels;

		public Tools(DataTools dataTools) {
			this.dataTools = dataTools;
			
			this.tokenSpanExtractors = new HashMap<String, TokenSpanExtractor<D, L>>();
			this.stringExtractors = new HashMap<String, StringExtractor<D, L>>();
			this.labelMappings = new HashMap<String, LabelMapping<L>>();
			this.genericFeatures = new HashMap<String, Feature<D, L>>();
			this.genericModels = new HashMap<String, SupervisedModel<D, L>>();
			
			this.labelMappings.put("Identity", new LabelMapping<L>() {
				public String toString() {
					return "Identity";
				}
				
				@Override
				public L map(L label) {
					return label;
				}
			});
			
			this.genericFeatures.put("GazetteerContains", new FeatureGazetteerContains<D, L>());
			this.genericFeatures.put("GazetteerEditDistance", new FeatureGazetteerEditDistance<D, L>());
			this.genericFeatures.put("GazetteerInitialism", new FeatureGazetteerInitialism<D, L>());
			this.genericFeatures.put("GazetteerPrefixTokens", new FeatureGazetteerPrefixTokens<D, L>());
			this.genericFeatures.put("NGramContext", new FeatureNGramContext<D, L>());
			this.genericFeatures.put("NGramSentence", new FeatureNGramSentence<D, L>());
			this.genericFeatures.put("NGramDep", new FeatureNGramDep<D, L>());
			
			this.genericModels.put("Creg", new SupervisedModelCreg<D, L>());
		}
		
		public DataTools getDataTools() {
			return this.dataTools;
		}
		
		public TokenSpanExtractor<D, L> getTokenSpanExtractor(String name) {
			return this.tokenSpanExtractors.get(name);
		}
		
		public StringExtractor<D, L> getStringExtractor(String name) {
			return this.stringExtractors.get(name);
		}
		
		public LabelMapping<L> getLabelMapping(String name) {
			return this.labelMappings.get(name);
		}
		
		public Feature<D, L> makeFeatureInstance(String genericFeatureName) {
			return this.genericFeatures.get(genericFeatureName).clone(this);
		}
		
		public SupervisedModel<D, L> makeModelInstance(String genericModelName) {
			return this.genericModels.get(genericModelName).clone(this);
		}
		
		public boolean addTokenSpanExtractor(TokenSpanExtractor<D, L> tokenSpanExtractor) {
			this.tokenSpanExtractors.put(tokenSpanExtractor.toString(), tokenSpanExtractor);
			return true;
		}
		
		public boolean addStringExtractor(StringExtractor<D, L> stringExtractor) {
			this.stringExtractors.put(stringExtractor.toString(), stringExtractor);
			return true;
		}
		
		public boolean addLabelMapping(LabelMapping<L> labelMapping) {
			this.labelMappings.put(labelMapping.toString(), labelMapping);
			return true;
		}
		
		public abstract L labelFromString(String str);
	}
}
