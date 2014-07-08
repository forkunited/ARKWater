package ark.data.annotation;

import java.util.HashMap;
import java.util.Map;

import ark.data.DataTools;
import ark.data.annotation.nlp.TokenSpan;
import ark.data.annotation.structure.DatumStructureCollection;
import ark.data.feature.Feature;
import ark.data.feature.FeatureConjunction;
import ark.data.feature.FeatureConstituencyPath;
import ark.data.feature.FeatureGazetteerContains;
import ark.data.feature.FeatureGazetteerEditDistance;
import ark.data.feature.FeatureGazetteerInitialism;
import ark.data.feature.FeatureGazetteerPrefixTokens;
import ark.data.feature.FeatureIdentity;
import ark.data.feature.FeatureDependencyPath;
import ark.data.feature.FeatureNGramContext;
import ark.data.feature.FeatureNGramDep;
import ark.data.feature.FeatureNGramSentence;
import ark.data.feature.FeatureNGramPoS;
import ark.model.SupervisedModel;
import ark.model.SupervisedModelCreg;
import ark.model.SupervisedModelLabelDistribution;
import ark.model.SupervisedModelPartition;
import ark.model.SupervisedModelSVM;
import ark.model.SupervisedModelSVMStructured;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.model.evaluation.metric.SupervisedModelEvaluationAccuracy;
import ark.model.evaluation.metric.SupervisedModelEvaluationF;
import ark.model.evaluation.metric.SupervisedModelEvaluationPrecision;
import ark.model.evaluation.metric.SupervisedModelEvaluationRecall;

/**
 * Datum represents a (possibly) labeled datum (training/evaluation
 * example).
 * 
 * @author Bill McDowell
 *
 * @param <L> label type
 */
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
	
	/**
	 * Tools contains tools for working with a particular type
	 * of datum.  For example, each type of datum type has a 
	 * collection of "extractors" for retrieving associated strings or 
	 * token spans that are necessary for computing features, and 
	 * each datum type also has an associated set of models and
	 * features that can be used with them.
	 * 
	 * @author Bill McDowell
	 *
	 * @param <D> datum type
	 * @param <L> label type
	 * 
	 */
	public static abstract class Tools<D extends Datum<L>, L> {
		public static interface StringExtractor<D extends Datum<L>, L> {
			String toString();
			String[] extract(D datum);
		}
		
		public static interface TokenSpanExtractor<D extends Datum<L>, L> {
			String toString();
			TokenSpan[] extract(D datum);
		}
		
		public static interface DoubleExtractor<D extends Datum<L>, L> {
			String toString();
			double[] extract(D datum);
		}
		
		public static interface LabelMapping<L> {
			String toString();
			L map(L label);
		}
		
		protected DataTools dataTools;
		
		private Map<String, TokenSpanExtractor<D, L>> tokenSpanExtractors;
		private Map<String, StringExtractor<D, L>> stringExtractors;
		private Map<String, DoubleExtractor<D, L>> doubleExtractors;
		private Map<String, LabelMapping<L>> labelMappings;
		
		private Map<String, Feature<D, L>> genericFeatures;
		private Map<String, SupervisedModel<D, L>> genericModels;
		private Map<String, SupervisedModelEvaluation<D, L>> genericEvaluations;

		private Map<String, DatumStructureCollection<D, L>> genericDatumStructureCollections;
		
		public Tools(DataTools dataTools) {
			this.dataTools = dataTools;
			
			this.tokenSpanExtractors = new HashMap<String, TokenSpanExtractor<D, L>>();
			this.stringExtractors = new HashMap<String, StringExtractor<D, L>>();
			this.doubleExtractors = new HashMap<String, DoubleExtractor<D, L>>();
			this.labelMappings = new HashMap<String, LabelMapping<L>>();
			this.genericFeatures = new HashMap<String, Feature<D, L>>();
			this.genericModels = new HashMap<String, SupervisedModel<D, L>>();
			this.genericEvaluations = new HashMap<String, SupervisedModelEvaluation<D, L>>();
			
			this.genericDatumStructureCollections = new HashMap<String, DatumStructureCollection<D, L>>();
			
			addLabelMapping(new LabelMapping<L>() {
				public String toString() {
					return "Identity";
				}
				
				@Override
				public L map(L label) {
					return label;
				}
			});
			
			
			addGenericFeature(new FeatureGazetteerContains<D, L>());
			addGenericFeature(new FeatureGazetteerEditDistance<D, L>());
			addGenericFeature(new FeatureGazetteerInitialism<D, L>());
			addGenericFeature(new FeatureGazetteerPrefixTokens<D, L>());
			addGenericFeature(new FeatureNGramContext<D, L>());
			addGenericFeature(new FeatureNGramSentence<D, L>());
			addGenericFeature(new FeatureNGramDep<D, L>());
			addGenericFeature(new FeatureIdentity<D, L>());
			addGenericFeature(new FeatureNGramPoS<D, L>());
			addGenericFeature(new FeatureDependencyPath<D, L>());
			addGenericFeature(new FeatureConstituencyPath<D, L>());
			addGenericFeature(new FeatureConjunction<D, L>());
			
			addGenericModel(new SupervisedModelCreg<D, L>());
			addGenericModel(new SupervisedModelLabelDistribution<D, L>());
			addGenericModel(new SupervisedModelSVM<D, L>());
			addGenericModel(new SupervisedModelSVMStructured<D, L>());
			addGenericModel(new SupervisedModelPartition<D, L>());
			
			addGenericEvaluation(new SupervisedModelEvaluationAccuracy<D, L>());
			addGenericEvaluation(new SupervisedModelEvaluationPrecision<D, L>());
			addGenericEvaluation(new SupervisedModelEvaluationRecall<D, L>());
			addGenericEvaluation(new SupervisedModelEvaluationF<D, L>());
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
		
		public DoubleExtractor<D, L> getDoubleExtractor(String name) {
			return this.doubleExtractors.get(name);
		}
		
		public LabelMapping<L> getLabelMapping(String name) {
			return this.labelMappings.get(name);
		}
		
		public Feature<D, L> makeFeatureInstance(String genericFeatureName) {
			return this.genericFeatures.get(genericFeatureName).clone(this, this.dataTools.getParameterEnvironment());
		}
		
		public SupervisedModel<D, L> makeModelInstance(String genericModelName) {
			return this.genericModels.get(genericModelName).clone(this, this.dataTools.getParameterEnvironment());
		}
		
		public SupervisedModelEvaluation<D, L> makeEvaluationInstance(String genericEvaluationName) {
			return this.genericEvaluations.get(genericEvaluationName).clone(this, this.dataTools.getParameterEnvironment());
		}
		
		public DatumStructureCollection<D, L> makeDatumStructureCollection(String genericCollectionName, DataSet<D, L> data) {
			return this.genericDatumStructureCollections.get(genericCollectionName).makeInstance(data);
		}
		
		public boolean addTokenSpanExtractor(TokenSpanExtractor<D, L> tokenSpanExtractor) {
			this.tokenSpanExtractors.put(tokenSpanExtractor.toString(), tokenSpanExtractor);
			return true;
		}
		
		public boolean addStringExtractor(StringExtractor<D, L> stringExtractor) {
			this.stringExtractors.put(stringExtractor.toString(), stringExtractor);
			return true;
		}
		
		public boolean addDoubleExtractor(DoubleExtractor<D, L> doubleExtractor) {
			this.doubleExtractors.put(doubleExtractor.toString(), doubleExtractor);
			return true;
		}
		
		public boolean addLabelMapping(LabelMapping<L> labelMapping) {
			this.labelMappings.put(labelMapping.toString(), labelMapping);
			return true;
		}
		
		public boolean addGenericFeature(Feature<D, L> feature) {
			this.genericFeatures.put(feature.getGenericName(), feature);
			return true;
		}
		
		public boolean addGenericModel(SupervisedModel<D, L> model) {
			this.genericModels.put(model.getGenericName(), model);
			return true;
		}
		
		public boolean addGenericEvaluation(SupervisedModelEvaluation<D, L> evaluation) {
			this.genericEvaluations.put(evaluation.getGenericName(), evaluation);
			return true;
		}
		
		public boolean addGenericDatumStructureCollection(DatumStructureCollection<D, L> datumStructureCollection) {
			this.genericDatumStructureCollections.put(datumStructureCollection.getGenericName(), datumStructureCollection);
			return true;
		}
		
		public abstract L labelFromString(String str);
	}
}
