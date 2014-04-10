package ark.data.annotation;

import java.util.HashMap;
import java.util.Map;

import ark.data.DataTools;
import ark.data.annotation.nlp.TokenSpan;
import ark.data.feature.Feature;
import ark.data.feature.FeatureConjunction;
import ark.data.feature.FeatureGazetteerContains;
import ark.data.feature.FeatureGazetteerEditDistance;
import ark.data.feature.FeatureGazetteerInitialism;
import ark.data.feature.FeatureGazetteerPrefixTokens;
import ark.data.feature.FeatureIdentity;
import ark.data.feature.FeatureLabeledDependencyPath;
import ark.data.feature.FeatureNGramContext;
import ark.data.feature.FeatureNGramDep;
import ark.data.feature.FeatureNGramSentence;
import ark.data.feature.FeatureNGramPoS;
import ark.model.SupervisedModel;
import ark.model.SupervisedModelCreg;
import ark.model.SupervisedModelLabelDistribution;
import ark.model.SupervisedModelSVMCostLearner;
import ark.model.cost.FactoredCost;
import ark.model.cost.FactoredCostConstant;
import ark.model.cost.FactoredCostLabel;
import ark.model.cost.FactoredCostLabelPair;
import ark.model.cost.FactoredCostLabelPairUnordered;
import ark.model.evaluation.metric.ClassificationEvaluation;
import ark.model.evaluation.metric.ClassificationEvaluationAccuracy;
import ark.model.evaluation.metric.ClassificationEvaluationF;
import ark.model.evaluation.metric.ClassificationEvaluationPrecision;
import ark.model.evaluation.metric.ClassificationEvaluationRecall;

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
		
		public static interface DoubleExtractor<D extends Datum<L>, L> {
			String toString();
			double[] extract(D datum);
		}
		
		public static interface LabelMapping<L> {
			String toString();
			L map(L label);
		}
		
		protected DataTools dataTools;
		
		protected Map<String, TokenSpanExtractor<D, L>> tokenSpanExtractors;
		protected Map<String, StringExtractor<D, L>> stringExtractors;
		protected Map<String, DoubleExtractor<D, L>> doubleExtractors;
		protected Map<String, LabelMapping<L>> labelMappings;
		
		protected Map<String, Feature<D, L>> genericFeatures;
		protected Map<String, SupervisedModel<D, L>> genericModels;
		protected Map<String, ClassificationEvaluation<D, L>> genericEvaluations;
		protected Map<String, FactoredCost<D, L>> genericFactoredCosts;

		public Tools(DataTools dataTools) {
			this.dataTools = dataTools;
			
			this.tokenSpanExtractors = new HashMap<String, TokenSpanExtractor<D, L>>();
			this.stringExtractors = new HashMap<String, StringExtractor<D, L>>();
			this.doubleExtractors = new HashMap<String, DoubleExtractor<D, L>>();
			this.labelMappings = new HashMap<String, LabelMapping<L>>();
			this.genericFeatures = new HashMap<String, Feature<D, L>>();
			this.genericModels = new HashMap<String, SupervisedModel<D, L>>();
			this.genericEvaluations = new HashMap<String, ClassificationEvaluation<D, L>>();
			this.genericFactoredCosts = new HashMap<String, FactoredCost<D, L>>();
			
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
			this.genericFeatures.put("Identity", new FeatureIdentity<D, L>());
			this.genericFeatures.put("NGramPoS", new FeatureNGramPoS<D, L>());
			this.genericFeatures.put("LabeledDependencyPath", new FeatureLabeledDependencyPath<D, L>());
			this.genericFeatures.put("Conjunction", new FeatureConjunction<D, L>());
			
			this.genericModels.put("Creg", new SupervisedModelCreg<D, L>());
			this.genericModels.put("LabelDistribution", new SupervisedModelLabelDistribution<D, L>());
			this.genericModels.put("SVMCostLearner", new SupervisedModelSVMCostLearner<D, L>());
			
			this.genericEvaluations.put("Accuracy", new ClassificationEvaluationAccuracy<D, L>());
			this.genericEvaluations.put("Precision", new ClassificationEvaluationPrecision<D, L>());
			this.genericEvaluations.put("Recall", new ClassificationEvaluationRecall<D, L>());
			this.genericEvaluations.put("F", new ClassificationEvaluationF<D, L>());
			
			this.genericFactoredCosts.put("Constant", new FactoredCostConstant<D, L>());
			this.genericFactoredCosts.put("Label", new FactoredCostLabel<D, L>());
			this.genericFactoredCosts.put("LabelPair", new FactoredCostLabelPair<D, L>());
			this.genericFactoredCosts.put("LabelPairUnordered", new FactoredCostLabelPairUnordered<D, L>());
			
			
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
		
		public ClassificationEvaluation<D, L> makeEvaluationInstance(String genericEvaluationName) {
			return this.genericEvaluations.get(genericEvaluationName).clone(this, this.dataTools.getParameterEnvironment());
		}
		
		public FactoredCost<D, L> makeFactoredCostInstance(String genericFactoredCostName) {
			return this.genericFactoredCosts.get(genericFactoredCostName).clone(this, this.dataTools.getParameterEnvironment());
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
		
		public abstract L labelFromString(String str);
	}
}
