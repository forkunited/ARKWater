package ark.model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.annotation.structure.DatumStructure;
import ark.data.annotation.structure.DatumStructureCollection;
import ark.data.feature.FeaturizedDataSet;
import ark.util.Pair;
import ark.util.SerializationUtil;

public class SupervisedModelSVMStructuredDeadPegasos<D extends Datum<L>, L> extends SupervisedModelSVMDeadPegasos<D, L> {
	protected String datumStructureOptimizer;
	protected String datumStructureCollection;
	protected DatumStructureCollection<D, L> trainingDatumStructureCollection;
	protected boolean includeStructuredTraining;
	
	public SupervisedModelSVMStructuredDeadPegasos() {
		super();
	}
	
	@Override
	protected boolean deserializeExtraInfo(String name, BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		if (name.equals("datumStructureCollection")) {
			this.datumStructureCollection = SerializationUtil.deserializeAssignmentRight(reader);
		} else if (name.equals("datumStructureOptimizer")) {
			this.datumStructureOptimizer = SerializationUtil.deserializeAssignmentRight(reader);
		} else if (name.equals("includeStructuredTraining")) {
			this.includeStructuredTraining = Boolean.valueOf(SerializationUtil.deserializeAssignmentRight(reader));
		} else {
			return super.deserializeExtraInfo(name, reader, datumTools);
		}
		
		return true;
	}

	@Override
	protected boolean serializeExtraInfo(Writer writer) throws IOException {		
		if (this.datumStructureCollection != null) {
			writer.write("\t");
			Pair<String, String> datumStructureCollectionAssignment = new Pair<String, String>("datumStructureCollection", this.datumStructureCollection);
			if (!SerializationUtil.serializeAssignment(datumStructureCollectionAssignment, writer))
				return false;
			writer.write("\n");
		}
		
		if (this.datumStructureOptimizer != null) {
			writer.write("\t");
			Pair<String, String> datumStructureOptimizerAssignment = new Pair<String, String>("datumStructureOptimizer", this.datumStructureOptimizer);
			if (!SerializationUtil.serializeAssignment(datumStructureOptimizerAssignment, writer))
				return false;
			writer.write("\n");
		}
		
		writer.write("\t");
		Pair<String, String> datumStructureOptimizerAssignment = new Pair<String, String>("includeStructuredTraining", String.valueOf(this.includeStructuredTraining));
		if (!SerializationUtil.serializeAssignment(datumStructureOptimizerAssignment, writer))
			return false;
		writer.write("\n");
		
		return super.serializeExtraInfo(writer);
	}
	
	@Override
	protected boolean initializeTraining(FeaturizedDataSet<D, L> data) {
		if (!super.initializeTraining(data))
			return false;
	
		if (this.includeStructuredTraining)
			this.trainingDatumStructureCollection = data.getDatumTools().makeDatumStructureCollection(this.datumStructureCollection, data);

		return true;
	}
	
	@Override
	protected boolean trainOneIteration(int iteration, FeaturizedDataSet<D, L> data) {
		if (!this.includeStructuredTraining)
			return super.trainOneIteration(iteration, data);
		
		List<Integer> dataPermutation = this.trainingDatumStructureCollection.constructRandomDatumStructurePermutation(this.random);
		for (Integer datumStructureIndex : dataPermutation) {
			DatumStructure<D, L> datumStructure = this.trainingDatumStructureCollection.getDatumStructure(datumStructureIndex);
			Map<D, Map<L, Double>> scoredDatumLabels = scoreDatumStructureLabels(data, datumStructure, true);
			Map<D, L> datumLabels = datumStructure.getDatumLabels(this.labelMapping);
			// Maybe just optimize here...?
			Map<D, L> bestDatumLabels = getBestDatumLabels(data, datumStructure, scoredDatumLabels);

			Map<Integer, Double> datumStructureFeatureValues = computeDatumStructureFeatureValues(data, datumStructure, datumLabels, iteration == 0);
			Map<Integer, Double> bestStructureFeatureValues = computeDatumStructureFeatureValues(data, datumStructure, bestDatumLabels, false);
			
			double eta = 1.0/(this.l2*this.t); // Learning rate
			this.s = (this.t > 1) ? (1.0-eta*this.l2)*this.s : 1; // Weight scalar
			
			// Update feature weights
			for (Entry<Integer, Double> featureValue : datumStructureFeatureValues.entrySet()) {	
				int weightIndex = featureValue.getKey();
				this.feature_W[weightIndex] += eta*featureValue.getValue()/this.s;		
			}
			
			for (Entry<Integer, Double> bestFeatureValue : bestStructureFeatureValues.entrySet()) {
				int weightIndex = bestFeatureValue.getKey();
				this.feature_W[weightIndex] -= eta*bestFeatureValue.getValue()/this.s;
			}
			
			// Update label biases
			for (int i = 0; i < this.bias_b.length; i++) {
				L label = this.labelIndices.reverseGet(i);
				int datumLabelCount = getLabelCount(datumLabels, label);
				int bestLabelCount = getLabelCount(bestDatumLabels, label);
				this.bias_g[i] = -datumLabelCount + bestLabelCount;
				this.bias_b[i] -= this.bias_g[i]*eta;
			}
			
			this.t++;
		}
		
		return true;
	}
	
	@Override
	protected double objectiveValue(FeaturizedDataSet<D, L> data) {
		if (!this.includeStructuredTraining)
			return super.objectiveValue(data);
		
		double value = 0.0;
		
		if (this.l2 > 0) {
			double l2Norm = 0;
			for (double W : this.feature_W)
				l2Norm += W*W*this.s*this.s;
			value += l2Norm*this.l2*.5;
		}
		
		// NOTE: This assumes that this function will only be called from training
		for (DatumStructure<D, L> datumStructure : this.trainingDatumStructureCollection) {
			Map<D, Map<L, Double>> scoredDatumLabels = scoreDatumStructureLabels(data, datumStructure, true);
			Map<D, L> datumLabels = datumStructure.getDatumLabels(this.labelMapping);
			Map<D, L> bestDatumLabels = getBestDatumLabels(data, datumStructure, scoredDatumLabels);
		
			double datumStructureScore = scoreDatumStructure(data, datumStructure, datumLabels, false);
			double bestStructureScore = scoreDatumStructure(data, datumStructure, bestDatumLabels, true);
		
			value += bestStructureScore - datumStructureScore;
		}
		
		return value;
	}
	
	public SupervisedModel<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		SupervisedModelSVMStructuredDeadPegasos<D, L> clone = (SupervisedModelSVMStructuredDeadPegasos<D, L>)super.clone(datumTools, environment);
		
		clone.labelIndices = this.labelIndices;
		clone.trainingIterations = this.trainingIterations;
		clone.datumStructureCollection = this.datumStructureCollection;
		clone.datumStructureOptimizer = this.datumStructureOptimizer;
		clone.includeStructuredTraining = this.includeStructuredTraining;
		
		return clone;
	}
	
	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelSVMStructuredDeadPegasos<D, L>();
	}

	@Override
	public String getGenericName() {
		return "SVMStructured";
	}
	
	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> posteriors = null;

		if (this.includeStructuredTraining) {
			posteriors = posteriorFromStructureScores(data);
		} else {
			posteriors = posteriorFromDatumScores(data);
		}

		return posteriors;
	}
	
	protected Map<D, Map<L, Double>> posteriorFromStructureScores(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> posteriors = new HashMap<D, Map<L, Double>>(data.size());
		DatumStructureCollection<D, L> datumStructureCollection = data.getDatumTools().makeDatumStructureCollection(this.datumStructureCollection, data);
		Map<D, L> bestDatumLabels = new HashMap<D, L>();

		for (DatumStructure<D, L> datumStructure : datumStructureCollection) {
			Map<D, Map<L, Double>> scoredDatumLabels = scoreDatumStructureLabels(data, datumStructure, false);
			bestDatumLabels.putAll(
				datumStructure.optimize(this.datumStructureOptimizer, scoredDatumLabels, this.fixedDatumLabels, this.validLabels, this.labelMapping)
			);
		}
		
		for (D datum : data) {
			posteriors.put(datum, new HashMap<L, Double>());
			L bestLabel = bestDatumLabels.get(datum);
			
			if (bestLabel == null) {
				double p = 1.0/this.validLabels.size();
				posteriors.get(datum).put(bestLabel, p);
			} else {
				for (L label : this.validLabels) {
					if (label.equals(bestLabel))
						posteriors.get(datum).put(label, 1.0);
					else
						posteriors.get(datum).put(label, 0.0);
				}
			}
		}
		
		return posteriors;
	}
	
	protected Map<D, Map<L, Double>> posteriorFromDatumScores(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> datumPosteriors = new HashMap<D, Map<L, Double>>(data.size());

		for (D datum : data) {
			datumPosteriors.put(datum, posteriorForDatum(data, datum));
		}
		
		DatumStructureCollection<D, L> datumStructureCollection = data.getDatumTools().makeDatumStructureCollection(this.datumStructureCollection, data);
		Map<D, Map<L, Double>> structurePosteriors = new HashMap<D, Map<L, Double>>(data.size());
		
		for (DatumStructure<D, L> datumStructure : datumStructureCollection) {
			Map<D, L> optimizedDatumLabels = datumStructure.optimize(this.datumStructureOptimizer, datumPosteriors, this.fixedDatumLabels, this.validLabels, this.labelMapping);
			for (Entry<D, L> entry : optimizedDatumLabels.entrySet()) {
				Map<L, Double> p = new HashMap<L, Double>();
				for (L validLabel : this.validLabels) {
					p.put(validLabel, 0.0);
				}
				p.put(entry.getValue(), 1.0);
				
				structurePosteriors.put(entry.getKey(), p);
			}
		}

		return structurePosteriors;
	}
	
	@Override
	public Map<D, L> classify(FeaturizedDataSet<D, L> data) {
		Map<D, L> classifiedData = null;

		if (this.includeStructuredTraining) {
			classifiedData = classifyFromStructureScores(data);
		} else {
			classifiedData = classifyFromDatumScores(data);
		}

		return classifiedData;
	}
	
	protected Map<D, L> classifyFromStructureScores(FeaturizedDataSet<D, L> data) {
		Map<D, L> classifiedData = new HashMap<D, L>(data.size());
		DatumStructureCollection<D, L> datumStructureCollection = data.getDatumTools().makeDatumStructureCollection(this.datumStructureCollection, data);
		Map<D, L> bestDatumLabels = new HashMap<D, L>();

		for (DatumStructure<D, L> datumStructure : datumStructureCollection) {
			Map<D, Map<L, Double>> scoredDatumLabels = scoreDatumStructureLabels(data, datumStructure, false);
			bestDatumLabels.putAll(
				datumStructure.optimize(this.datumStructureOptimizer, scoredDatumLabels, this.fixedDatumLabels, this.validLabels, this.labelMapping)
			);
		}
		
		for (D datum : data) {
			L bestLabel = bestDatumLabels.get(datum);
			classifiedData.put(datum, (bestLabel == null) ? this.labelIndices.reverseGet(0) : bestLabel);
		}
		
		return classifiedData;
	}
	
	protected Map<D, L> classifyFromDatumScores(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> datumPosteriors = new HashMap<D, Map<L, Double>>(data.size());
		Map<D, L> classifiedData = new HashMap<D, L>(data.size());
		
		for (D datum : data) {
			datumPosteriors.put(datum, posteriorForDatum(data, datum));
		}
		
		DatumStructureCollection<D, L> datumStructureCollection = data.getDatumTools().makeDatumStructureCollection(this.datumStructureCollection, data);
		for (DatumStructure<D, L> datumStructure : datumStructureCollection) {
			Map<D, L> optimizedDatumLabels = datumStructure.optimize(this.datumStructureOptimizer, datumPosteriors, this.fixedDatumLabels, this.validLabels, this.labelMapping);
			classifiedData.putAll(optimizedDatumLabels);
		}

		return classifiedData;
	}
	
	protected int getLabelCount(Map<D, L> datumsToLabels, L countLabel) {
		int count = 0;
		for (L label : datumsToLabels.values())
			if (label.equals(countLabel))
				count++;
		return count;
	}
	
	protected Map<D, Map<L, Double>> scoreDatumStructureLabels(FeaturizedDataSet<D, L> data, DatumStructure<D, L> datumStructure, boolean includeCost) {
		Map<D, Map<L, Double>> datumLabelScores = new HashMap<D, Map<L, Double>>();
		
		for (D datum : datumStructure) {
			Map<L, Double> scores = new HashMap<L, Double>();
			
			double max = Double.NEGATIVE_INFINITY;
			for (L label : this.validLabels) {
				double score = scoreLabel(data, datum, label, includeCost);
				scores.put(label, score);
				if (score > max)
					max = score;
			}
			
			datumLabelScores.put(datum, scores);
		}
		
		return datumLabelScores;
	}
	
	protected double scoreDatumStructure(FeaturizedDataSet<D, L> data, DatumStructure<D, L> datumStructure, Map<D, L> structureLabels, boolean includeCost) {
		double score = 0.0;
		
		Map<Integer, Double> datumStructureFeatureValues = computeDatumStructureFeatureValues(data, datumStructure, structureLabels, false);
		for (Entry<Integer, Double> entry : datumStructureFeatureValues.entrySet()) {
			score += this.s*this.feature_W[entry.getKey()]*entry.getValue();
		}
		
		for (int i = 0; i < this.bias_b.length; i++) {
			L label = this.labelIndices.reverseGet(i);
			int datumLabelCount = getLabelCount(structureLabels, label);
			score += this.bias_b[i]*datumLabelCount;
		}
		
		if (includeCost) {
			for (D datum : datumStructure) {
				if (!mapValidLabel(datum.getLabel()).equals(structureLabels.get(datum)))
					score += 1.0;
			}
		}
		
		return score;
	}
	
	protected Map<Integer, Double> computeDatumStructureFeatureValues(FeaturizedDataSet<D,L> data, DatumStructure<D, L> datumStructure, Map<D, L> structureLabels, boolean cacheFeatureNames) {
		Map<Integer, Double> featureValues = new HashMap<Integer, Double>();
		int numDatumFeatures = data.getFeatureVocabularySize();
		for (D datum : datumStructure) {
			Map<Integer, Double> datumFeatureValues = data.getFeatureVocabularyValues(datum);
			int labelIndex = this.labelIndices.get(structureLabels.get(datum));
			int featureLabelOffset = numDatumFeatures*labelIndex;
			
			for (Entry<Integer, Double> entry : datumFeatureValues.entrySet()) {
				int featureIndex = featureLabelOffset + entry.getKey();
				if (!featureValues.containsKey(featureIndex))
					featureValues.put(featureIndex, 0.0);
				featureValues.put(featureIndex, featureValues.get(featureIndex) + entry.getValue());
			}
			
			if (cacheFeatureNames) {
				List<Integer> missingNameKeys = new ArrayList<Integer>();
				for (Integer key : datumFeatureValues.keySet())
					if (!this.featureNames.containsKey(key))
						missingNameKeys.add(key);
				this.featureNames.putAll(data.getFeatureVocabularyNamesForIndices(missingNameKeys));				
			}
		}
		
		return featureValues;
	}
	
	protected Map<D, L> getBestDatumLabels(FeaturizedDataSet<D, L> data, DatumStructure<D, L> datumStructure, Map<D, Map<L, Double>> scoredDatumLabels) {
		Map<D, L> optimizedDatumLabels = datumStructure.optimize(this.datumStructureOptimizer, scoredDatumLabels, this.fixedDatumLabels, this.validLabels, this.labelMapping);
		Map<D, L> actualDatumLabels = datumStructure.getDatumLabels(this.labelMapping);
		
		double optimizedScore = scoreDatumStructure(data, datumStructure, optimizedDatumLabels, true);
		double actualScore = scoreDatumStructure(data, datumStructure, actualDatumLabels, false);
		
		if (actualScore > optimizedScore)
			return actualDatumLabels;
		else
			return optimizedDatumLabels;
	}
}
