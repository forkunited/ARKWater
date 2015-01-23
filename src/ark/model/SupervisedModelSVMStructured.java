/**
 * Copyright 2014 Bill McDowell 
 *
 * This file is part of theMess (https://github.com/forkunited/theMess)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy 
 * of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the 
 * License for the specific language governing permissions and limitations 
 * under the License.
 */

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
import ark.util.BidirectionalLookupTable;
import ark.util.Pair;
import ark.util.SerializationUtil;

/**
 * SupervisedModelSVMStructured represents a structured SVM trained with
 * SGD using AdaGrad to determine the learning rate.  The AdaGrad minimization
 * uses sparse updates based on the loss gradients and 'occasional updates'
 * for the regularizer.  It's unclear whether the occasional regularizer
 * gradient updates are theoretically sound when used with AdaGrad (haven't
 * taken the time to think about it), but it seems to work anyway.
 * 
 * The structured SVM is described in some detail in the 
 * papers/TemporalOrderingNotes.pdf document in the TemporalOrdering
 * repository at https://github.com/forkunited/TemporalOrdering.
 * 
 * If 'includeStructuredTraining' is set to false, then the model is trained
 * as a non-structured SVM, and the structure is only imposed at inference 
 * time using the datum label scores to optimize the structure.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public class SupervisedModelSVMStructured<D extends Datum<L>, L> extends SupervisedModelSVM<D, L> {
	protected String datumStructureOptimizer; // Name of the optimizer stored in Datum.Tools to use for label inference
	protected String datumStructureCollection; // Name of the datum structure collection in Datum.Tools to use to create DatumStructures
	protected DatumStructureCollection<D, L> trainingDatumStructureCollection; // DatumStructureCollection instantiated for the training data
	protected boolean includeStructuredTraining; // Indicates whether or not to include structure in training
	
	public SupervisedModelSVMStructured() {
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
	
		// If structure in training, then partition the data set into
		// datum structures (ark.data.structure.DatumStructure) using
		// the reference datum structure collection 
		// (ark.data.structure.DatumStructureCollection)
		if (this.includeStructuredTraining)
			this.trainingDatumStructureCollection = data.getDatumTools().makeDatumStructureCollection(this.datumStructureCollection, data);
		return true;
	}
	
	/**
	 * @param iteration
	 * @param data
	 * @return true if the model has been trained using one pass over the full training
	 * data set.
	 */
	@Override
	protected boolean trainOneIteration(int iteration, FeaturizedDataSet<D, L> data) {
		// If no structure in training, then train as unstructured SVM
		if (!this.includeStructuredTraining)
			return super.trainOneIteration(iteration, data);
		
		List<Integer> dataPermutation = this.trainingDatumStructureCollection.constructRandomDatumStructurePermutation(this.random);
		double N = dataPermutation.size();
		for (Integer datumStructureIndex : dataPermutation) {
			double K = N/4.0;
			boolean regularizerUpdate = (this.t % K == 0); // for "occasionality trick"
			
			DatumStructure<D, L> datumStructure = this.trainingDatumStructureCollection.getDatumStructure(datumStructureIndex);
			// Map datums to labels to their current scores
			Map<D, Map<L, Double>> scoredDatumLabels = scoreDatumStructureLabels(data, datumStructure, true);
			Map<D, L> datumLabels = datumStructure.getDatumLabels(this.labelMapping);
			
			// Best datum labels according to model's inference based on current weights
			Map<D, L> bestDatumLabels = getBestDatumLabels(data, datumStructure, scoredDatumLabels);

			Map<Integer, Double> datumStructureFeatureValues = computeDatumStructureFeatureValues(data, datumStructure, datumLabels, iteration == 0);
			Map<Integer, Double> bestStructureFeatureValues = computeDatumStructureFeatureValues(data, datumStructure, bestDatumLabels, false);
			
			// Update feature weight gradients
			Map<Integer, Double> gMap = new HashMap<Integer, Double>();
			
			for (Entry<Integer, Double> featureEntry : datumStructureFeatureValues.entrySet()) {
				int weightIndex = featureEntry.getKey();
				gMap.put(weightIndex, -featureEntry.getValue());
			}
			
			for (Entry<Integer, Double> featureEntry : bestStructureFeatureValues.entrySet()) {
				int weightIndex = featureEntry.getKey();
				if (gMap.containsKey(weightIndex))
					gMap.put(weightIndex, gMap.get(weightIndex) + featureEntry.getValue());
				else
					gMap.put(weightIndex, featureEntry.getValue());
			}
			
			// Occasionally (every K datums) include regularizer term in computation of feature weight gradients
			if (regularizerUpdate) {	
				for (Entry<Integer, Double> wEntry : this.feature_w.entrySet()) {
					if (!gMap.containsKey(wEntry.getKey()))
						gMap.put(wEntry.getKey(), (K/N)*this.l2*wEntry.getValue());
					else 
						gMap.put(wEntry.getKey(), gMap.get(wEntry.getKey()) + (K/N)*this.l2*wEntry.getValue());
				}
			}
				
			// Update feature weights based on computed gradients
			for (Entry<Integer, Double> gEntry : gMap.entrySet()) {
				int weightIndex = gEntry.getKey();
				double g = gEntry.getValue();
				
				if (g == 0)
					continue;
				
				if (!this.feature_w.containsKey(weightIndex)) {
					this.feature_w.put(weightIndex, 0.0);
					this.feature_G.put(weightIndex, 0.0);
				}
				
				double G = this.feature_G.get(weightIndex) + g*g;
				this.feature_G.put(weightIndex, G);
				
				double eta = 1.0/Math.sqrt(G);
				
				this.feature_w.put(weightIndex, this.feature_w.get(weightIndex) - eta*g);
			}
			
			// Update label biases
			for (int i = 0; i < this.bias_b.length; i++) {
				L label = this.labelIndices.reverseGet(i);
				int datumLabelCount = getLabelCount(datumLabels, label);
				int bestLabelCount = getLabelCount(bestDatumLabels, label);
				double g = -datumLabelCount + bestLabelCount;
				
				if (g == 0)
					continue;
				
				this.bias_G[i] += g*g;
				
				double eta = 1.0/Math.sqrt(this.bias_G[i]);
				
				this.bias_b[i] -= g*eta;
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
			for (double w : this.feature_w.values())
				l2Norm += w*w;
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
	
	@SuppressWarnings("unchecked")
	public <D1 extends Datum<L1>, L1> SupervisedModel<D1, L1> clone(Datum.Tools<D1, L1> datumTools, Map<String, String> environment, boolean copyLabelObjects) {
		SupervisedModelSVMStructured<D1, L1> clone = (SupervisedModelSVMStructured<D1, L1>)super.clone(datumTools, environment, copyLabelObjects);
		
		if (copyLabelObjects)
			clone.labelIndices = (BidirectionalLookupTable<L1, Integer>)this.labelIndices;
		
		
		clone.trainingIterations = this.trainingIterations;
		clone.datumStructureCollection = this.datumStructureCollection;
		clone.datumStructureOptimizer = this.datumStructureOptimizer;
		clone.includeStructuredTraining = this.includeStructuredTraining;
		
		return clone;
	}
	
	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelSVMStructured<D, L>();
	}

	@Override
	public String getGenericName() {
		return "SVMStructured";
	}
	
	/**
	 * @param data
	 * @return a map from datums in data to their posteriors.  The posteriors are
	 * computed differently depending on whether structure was imposed at training
	 * or not.  See methods below for details.
	 */
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
	
	/**
	 * @param data
	 * @return a map from datums in data to their posteriors based on optimal label structures
	 * learned through structured training. Labels in the optimal datum structures are given 
	 * a posterior value of 1, and all other labels are given a posterior value 0.
	 */
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
	
	/**
	 * @param data
	 * @return a map from datums in data to their posteriors based on label structures
	 * optimized using the individual datum posteriors.
	 * Labels in the optimal datum structures are given a posterior value of 1, and all other 
	 * labels are given a posterior value 0
	 */
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
	
	/**
	 * @param data
	 * @return a map from datums to predicted labels based on 
	 * optimal label structures. This method works analogously to the 
	 * posterior methods defined above.
	 */
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
			if (bestLabel == null)
				data.getDatumTools().getDataTools().getOutputWriter().debugWriteln("WARNING: Optimizer returned no label for datum " + datum.getId());
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
	
	/**
	 * @param data
	 * @param datumStructure
	 * @param includeCost
	 * @return a map from datums in datumStructure to scores of their
	 * label assignments.
	 */
	protected Map<D, Map<L, Double>> scoreDatumStructureLabels(FeaturizedDataSet<D, L> data, DatumStructure<D, L> datumStructure, boolean includeCost) {
		Map<D, Map<L, Double>> datumLabelScores = new HashMap<D, Map<L, Double>>();
		
		for (D datum : datumStructure) {
			Map<L, Double> scores = new HashMap<L, Double>();

			for (L label : this.validLabels) {
				double score = scoreLabel(data, datum, label, includeCost);
				scores.put(label, score);
			}
			
			datumLabelScores.put(datum, scores);
		}
		
		return datumLabelScores;
	}
	
	protected double scoreDatumStructure(FeaturizedDataSet<D, L> data, DatumStructure<D, L> datumStructure, Map<D, L> structureLabels, boolean includeCost) {
		double score = 0.0;
	
		Map<Integer, Double> datumStructureFeatureValues = computeDatumStructureFeatureValues(data, datumStructure, structureLabels, false);
		for (Entry<Integer, Double> entry : datumStructureFeatureValues.entrySet()) {
			if (this.feature_w.containsKey(entry.getKey()))
				score += this.feature_w.get(entry.getKey())*entry.getValue();
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
			Map<Integer, Double> datumFeatureValues = data.getFeatureVocabularyValuesAsMap(datum);
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
