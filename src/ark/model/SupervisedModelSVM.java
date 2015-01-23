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
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import java.util.Random;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.annotation.Datum.Tools.LabelMapping;
import ark.data.feature.FeaturizedDataSet;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.BidirectionalLookupTable;
import ark.util.OutputWriter;
import ark.util.Pair;
import ark.util.SerializationUtil;

/**
 * SupervisedModelSVM represents a multi-class SVM trained with
 * SGD using AdaGrad to determine the learning rate.  The AdaGrad minimization
 * uses sparse updates based on the loss gradients and 'occasional updates'
 * for the regularizer.  It's unclear whether the occasional regularizer
 * gradient updates are theoretically sound when used with AdaGrad (haven't
 * taken the time to think about it), but it seems to work anyway.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public class SupervisedModelSVM<D extends Datum<L>, L> extends SupervisedModel<D, L> {
	protected BidirectionalLookupTable<L, Integer> labelIndices;
	protected int trainingIterations; // number of training iterations for which to run (set through 'extra info')
	protected boolean earlyStopIfNoLabelChange; // whether to have early stopping when no prediction changes on dev set (set through 'extra info')
	protected Map<Integer, String> featureNames; // map from feature indices to their names
	protected int numFeatures; // total number of features
	protected double[] bias_b;
	protected Map<Integer, Double> feature_w; // Labels x (Input features (percepts)) sparse weights mapped from weight indices 
	
	// Adagrad stuff
	protected int t;
	protected Map<Integer, Double> feature_G;  // Just diagonal
	protected double[] bias_G;
	
	protected double l2; // l2 regularizer
	protected double epsilon = 0;
	protected String[] hyperParameterNames = { "l2", "epsilon" };
	
	protected Random random;

	public SupervisedModelSVM() {
		this.featureNames = new HashMap<Integer, String>();
	}
	
	@Override
	public boolean setLabels(Set<L> validLabels, LabelMapping<L> labelMapping) {
		if (!super.setLabels(validLabels, labelMapping))
			return false;
		
		setLabelIndices();
		
		return true;
	}
	
	protected boolean setLabelIndices() {
		this.labelIndices = new BidirectionalLookupTable<L, Integer>();
		int i = 0;
		for (L label : this.validLabels) {
			this.labelIndices.put(label, i);
			i++;
		}
		return true;
	}
	
	@Override
	protected boolean deserializeExtraInfo(String name, BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		if (this.validLabels != null && this.labelIndices == null) {
			// Might be better to do this somewhere else...?
			setLabelIndices();
		}
		
		if (name.equals("trainingIterations")) {
			this.trainingIterations = Integer.valueOf(SerializationUtil.deserializeAssignmentRight(reader));
		} else if (name.equals("earlyStopIfNoLabelChange")){
			this.earlyStopIfNoLabelChange = Boolean.valueOf(SerializationUtil.deserializeAssignmentRight(reader));
			System.out.println("\n\n\n");
			System.out.println("earlyStopIfnoLabelChange: " + earlyStopIfNoLabelChange);
			System.out.println("\n\n\n");
		}
		
		return true;
	}

	@Override
	protected boolean serializeExtraInfo(Writer writer) throws IOException {
		writer.write("\t");
		Pair<String, String> trainingIterationsAssignment = new Pair<String, String>("trainingIterations", String.valueOf(this.trainingIterations));
		if (!SerializationUtil.serializeAssignment(trainingIterationsAssignment, writer))
			return false;
		writer.write("\n");
		
		return true;
	}

	@Override
	public boolean train(FeaturizedDataSet<D, L> data, FeaturizedDataSet<D, L> testData, List<SupervisedModelEvaluation<D, L>> evaluations) {
		OutputWriter output = data.getDatumTools().getDataTools().getOutputWriter();
		
		if (!initializeTraining(data))
			return false;
		
		//double prevObjectiveValue = objectiveValue(data);
		Map<D, L> prevPredictions = classify(testData);
		List<Double> prevEvaluationValues = new ArrayList<Double>();
		for (SupervisedModelEvaluation<D, L> evaluation : evaluations) {
			prevEvaluationValues.add(evaluation.evaluate(this, testData, prevPredictions));
		}
		
		output.debugWriteln("Training " + getGenericName() + " for " + this.trainingIterations + " iterations...");
		
		for (int iteration = 0; iteration < this.trainingIterations; iteration++) {
			if (!trainOneIteration(iteration, data)) 
				return false;
			
			if (iteration % 10 == 0) {
				//double objectiveValue = objectiveValue(data);
				//double objectiveValueDiff = objectiveValue - prevObjectiveValue;
				Map<D, L> predictions = classify(testData);
				int labelDifferences = countLabelDifferences(prevPredictions, predictions);
				if (earlyStopIfNoLabelChange && labelDifferences == 0 && iteration > 10)
					break;
			
				List<Double> evaluationValues = new ArrayList<Double>();
				for (SupervisedModelEvaluation<D, L> evaluation : evaluations) {
					evaluationValues.add(evaluation.evaluate(this, testData, predictions));
				}
				
				String statusStr = "(l2=" + this.l2 + ") Finished iteration " + iteration + /*" objective diff: " + objectiveValueDiff + " objective: " + objectiveValue + */" prediction-diff: " + labelDifferences + "/" + predictions.size() + " ";
				for (int i = 0; i < evaluations.size(); i++) {
					String evaluationName = evaluations.get(i).toString(false);
					double evaluationDiff = evaluationValues.get(i) - prevEvaluationValues.get(i);
					statusStr += evaluationName + " diff: " + evaluationDiff + " " + evaluationName + ": " + evaluationValues.get(i) + " ";
				}
					
				output.debugWriteln(statusStr);
				
				/*
				if (iteration > 20 && Math.abs(objectiveValueDiff) < this.epsilon) {
					output.debugWriteln("(l2=" + this.l2 + ") Terminating early at iteration " + iteration);
					break;
				}*/
				
				// prevObjectiveValue = objectiveValue;
				prevPredictions = predictions;
				prevEvaluationValues = evaluationValues;
			} else {
				output.debugWriteln("(l2=" + this.l2 + ") Finished iteration " + iteration);
			}
		}
		
		return true;
	}
	
	protected boolean initializeTraining(FeaturizedDataSet<D, L> data) {
		if (this.feature_w == null) {
			this.t = 1;
			
			this.bias_b = new double[this.validLabels.size()];
			this.numFeatures = data.getFeatureVocabularySize();
			this.feature_w = new HashMap<Integer, Double>(); 	
	
			this.bias_G = new double[this.bias_b.length];
			this.feature_G = new HashMap<Integer, Double>();
		}
		
		this.random = data.getDatumTools().getDataTools().makeLocalRandom();
		
		return true;
	}
	
	/**
	 * @param iteration
	 * @param data
	 * @return true if the model has been trained for a full pass over the
	 * training data set
	 */
	protected boolean trainOneIteration(int iteration, FeaturizedDataSet<D, L> data) {
		List<Integer> dataPermutation = data.constructRandomDataPermutation(this.random);
		
		for (Integer datumId : dataPermutation) {
			D datum = data.getDatumById(datumId);
			L datumLabel = this.mapValidLabel(datum.getLabel());
			L bestLabel = argMaxScoreLabel(data, datum, true);

			if (!trainOneDatum(datum, datumLabel, bestLabel, iteration, data)) {
				return false;
			}
			
			this.t++;
		}
		return true;
	}
	
	/**
	 * 
	 * @param datum
	 * @param datumLabel
	 * @param bestLabel
	 * @param iteration
	 * @param data
	 * @return true if the model has made SGD weight updates from a single datum.
	 */
	protected boolean trainOneDatum(D datum, L datumLabel, L bestLabel, int iteration, FeaturizedDataSet<D, L> data) {
		int N = data.size();
		double K = N/4.0;
		boolean datumLabelBest = datumLabel.equals(bestLabel);
		boolean regularizerUpdate = (this.t % K == 0); // for "occasionality trick"
		
		Map<Integer, Double> datumFeatureValues = data.getFeatureVocabularyValuesAsMap(datum);
		
		if (iteration == 0) {
			List<Integer> missingNameKeys = new ArrayList<Integer>();
			for (Integer key : datumFeatureValues.keySet())
				if (!this.featureNames.containsKey(key))
					missingNameKeys.add(key);
			this.featureNames.putAll(data.getFeatureVocabularyNamesForIndices(missingNameKeys));
		}
		
		if (datumLabelBest && !regularizerUpdate) // No update necessary
			return true;
			
		// Update feature weights
		if (!regularizerUpdate) { // Update only for loss function gradients
			for (Entry<Integer, Double> featureValue : datumFeatureValues.entrySet()) {
				int i_datumLabelWeight = getWeightIndex(datumLabel, featureValue.getKey());
				int i_bestLabelWeight = getWeightIndex(bestLabel, featureValue.getKey());
				
				if (!this.feature_w.containsKey(i_datumLabelWeight)) {
					this.feature_w.put(i_datumLabelWeight, 0.0);
					this.feature_G.put(i_datumLabelWeight, 0.0);
				}
				
				if (!this.feature_w.containsKey(i_bestLabelWeight)) {
					this.feature_w.put(i_bestLabelWeight, 0.0);
					this.feature_G.put(i_bestLabelWeight, 0.0);
				}
				
				// Gradients
				double g_datumLabelWeight = -featureValue.getValue();
				double g_bestLabelWeight = featureValue.getValue();
				
				// Adagrad G
				double G_datumLabelWeight = this.feature_G.get(i_datumLabelWeight) + g_datumLabelWeight*g_datumLabelWeight;
				double G_bestLabelWeight = this.feature_G.get(i_bestLabelWeight) + g_bestLabelWeight*g_bestLabelWeight;
				
				this.feature_G.put(i_datumLabelWeight, G_datumLabelWeight);
				this.feature_G.put(i_bestLabelWeight, G_bestLabelWeight);
				
				// Learning rates
				double eta_datumLabelWeight = 1.0/Math.sqrt(G_datumLabelWeight);
				double eta_bestLabelWeight = 1.0/Math.sqrt(G_bestLabelWeight);
				
				// Weight update
				this.feature_w.put(i_datumLabelWeight, this.feature_w.get(i_datumLabelWeight) - eta_datumLabelWeight*g_datumLabelWeight);
				this.feature_w.put(i_bestLabelWeight, this.feature_w.get(i_bestLabelWeight) - eta_bestLabelWeight*g_bestLabelWeight);
			}
		} else { // Full weight update for regularizer
			Map<Integer, Double> g = new HashMap<Integer, Double>(); // gradients
			
			// Gradient update for hinge loss
			for (Entry<Integer, Double> featureValue : datumFeatureValues.entrySet()) {
				int i_datumLabelWeight = getWeightIndex(datumLabel, featureValue.getKey());
				int i_bestLabelWeight = getWeightIndex(bestLabel, featureValue.getKey());
				
				g.put(i_datumLabelWeight, -featureValue.getValue());
				g.put(i_bestLabelWeight, featureValue.getValue());
			}
			
			// Occasional gradient update for regularizer (this happens after every K training datum updates)
			for (Entry<Integer, Double> wEntry : this.feature_w.entrySet()) {
				if (!g.containsKey(wEntry.getKey()))
					g.put(wEntry.getKey(), (K/N)*this.l2*wEntry.getValue());
				else 
					g.put(wEntry.getKey(), g.get(wEntry.getKey()) + (K/N)*this.l2*wEntry.getValue());
			}
			
			// Update weights based on gradients
			for (Entry<Integer, Double> gEntry : g.entrySet()) {
				if (gEntry.getValue() == 0)
					continue;
				
				if (!this.feature_w.containsKey(gEntry.getKey())) {
					this.feature_w.put(gEntry.getKey(), 0.0);
					this.feature_G.put(gEntry.getKey(), 0.0);
				}
				
				// Adagrad G
				double G = this.feature_G.get(gEntry.getKey()) + gEntry.getValue()*gEntry.getValue();
				this.feature_G.put(gEntry.getKey(), G);
				
				double eta = 1.0/Math.sqrt(G);
				this.feature_w.put(gEntry.getKey(), this.feature_w.get(gEntry.getKey()) - eta*gEntry.getValue());
			}
		}
			
		// Update label biases
		for (int i = 0; i < this.bias_b.length; i++) {
			// Bias gradient based on hinge loss
			double g = ((this.labelIndices.get(datumLabel) == i) ? -1.0 : 0.0) +
							(this.labelIndices.get(bestLabel) == i ? 1.0 : 0.0);
			
			if (g == 0)
				continue;
			
			this.bias_G[i] += g*g;
			double eta = 1.0/Math.sqrt(this.bias_G[i]);
			this.bias_b[i] -= eta*g;
		}
		
		return true;
	}
	
	private int countLabelDifferences(Map<D, L> labels1, Map<D, L> labels2) {
		int count = 0;
		for (Entry<D, L> entry: labels1.entrySet()) {
			if (!labels2.containsKey(entry.getKey()) || !entry.getValue().equals(labels2.get(entry.getKey())))
				count++;
		}
		return count;
	}
	
	protected double objectiveValue(FeaturizedDataSet<D, L> data) {
		double value = 0;
		
		if (this.l2 > 0) {
			double l2Norm = 0;
			for (Entry<Integer, Double> wEntry : this.feature_w.entrySet())
				l2Norm += wEntry.getValue()*wEntry.getValue();
			value += l2Norm*this.l2*.5;
		}
		
		for (D datum : data) {
			double maxScore = maxScoreLabel(data, datum, true);
			double datumScore = scoreLabel(data, datum, datum.getLabel(), false);
			value += maxScore - datumScore;
		}
		
		return value;
	}
	
	protected double maxScoreLabel(FeaturizedDataSet<D, L> data, D datum, boolean includeCost) {
		double maxScore = Double.NEGATIVE_INFINITY;
		for (L label : this.validLabels) {
			double score = scoreLabel(data, datum, label, includeCost);
			if (score >= maxScore) {
				maxScore = score;
			}
		}
		return maxScore;
	}
	
	protected L argMaxScoreLabel(FeaturizedDataSet<D, L> data, D datum, boolean includeCost) {
		double maxScore = Double.NEGATIVE_INFINITY;
		List<L> maxLabels = null; // for breaking ties randomly
		L maxLabel = null;
		for (L label : this.validLabels) {
			double score = scoreLabel(data, datum, label, includeCost);
			
			if (score == maxScore) {
				if (maxLabels == null) {
					maxLabels = new ArrayList<L>();
					if (maxLabel != null) {
						maxLabels.add(maxLabel);
						maxLabel = null;
					}
				}
				maxLabels.add(label);
			} else if (score > maxScore) {
				maxScore = score;
				maxLabel = label;
				maxLabels = null;
			}
		}
		
		if (maxLabels != null)
			return maxLabels.get(this.random.nextInt(maxLabels.size()));
		else
			return maxLabel;
	}
	
	protected double scoreLabel(FeaturizedDataSet<D, L> data, D datum, L label, boolean includeCost) {
		double score = 0;		
		
		Map<Integer, Double> featureValues = data.getFeatureVocabularyValuesAsMap(datum);
		int labelIndex = this.labelIndices.get(label);
		for (Entry<Integer, Double> entry : featureValues.entrySet()) {
			int wIndex = this.getWeightIndex(label, entry.getKey());
			if (this.feature_w.containsKey(wIndex))
				score += this.feature_w.get(wIndex)*entry.getValue();
		}
		
		score += this.bias_b[labelIndex];

		if (includeCost) {
			if (!mapValidLabel(datum.getLabel()).equals(label))
				score += 1.0;
		}
		
		return score;
	}
	
	protected int getWeightIndex(L label, int featureIndex) {
		return this.labelIndices.get(label)*this.numFeatures + featureIndex;
	}
	
	protected int getWeightIndex(int labelIndex, int featureIndex) {
		return labelIndex*this.numFeatures + featureIndex;
	}
	
	protected int getFeatureIndex(int weightIndex) {
		return weightIndex % this.numFeatures;
	}
	
	protected int getLabelIndex(int weightIndex) {
		return weightIndex / this.numFeatures;
	}
	
	@Override
	public String[] getParameterNames() {
		return this.hyperParameterNames;
	}

	@Override
	public String getParameterValue(String parameter) {
		if (parameter.equals("l2"))
			return String.valueOf(this.l2);
		else if (parameter.equals("epsilon"))
			return String.valueOf(this.epsilon);
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("l2"))
			this.l2 = Double.valueOf(parameterValue);
		else if (parameter.equals("epsilon"))
			this.epsilon = Double.valueOf(parameterValue);
		else
			return false;
		return true;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public <D1 extends Datum<L1>, L1> SupervisedModel<D1, L1> clone(Datum.Tools<D1, L1> datumTools, Map<String, String> environment, boolean copyLabelObjects) {
		SupervisedModelSVM<D1, L1> clone = (SupervisedModelSVM<D1, L1>)super.clone(datumTools, environment, copyLabelObjects);
		
		if (copyLabelObjects)
			clone.labelIndices = (BidirectionalLookupTable<L1, Integer>)this.labelIndices;
		
		clone.trainingIterations = this.trainingIterations;
		clone.earlyStopIfNoLabelChange = this.earlyStopIfNoLabelChange;
		
		return clone;
	}
	
	@Override
	protected boolean deserializeParameters(BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		Pair<String, String> tAssign = SerializationUtil.deserializeAssignment(reader);
		Pair<String, String> numWeightsAssign = SerializationUtil.deserializeAssignment(reader);
	
		int numWeights = Integer.valueOf(numWeightsAssign.getSecond());
		this.numFeatures = numWeights / this.labelIndices.size();
		
		this.t = Integer.valueOf(tAssign.getSecond());
		this.featureNames = new HashMap<Integer, String>();
		
		this.feature_w = new HashMap<Integer, Double>();
		this.feature_G = new HashMap<Integer, Double>();
		
		this.bias_b = new double[this.labelIndices.size()];
		this.bias_G = new double[this.bias_b.length];	
		
		String assignmentLeft = null;
		while ((assignmentLeft = SerializationUtil.deserializeAssignmentLeft(reader)) != null) {
			if (assignmentLeft.equals("labelFeature")) {
				String labelFeature = SerializationUtil.deserializeGenericName(reader);
				Map<String, String> featureParameters = SerializationUtil.deserializeArguments(reader);
				
				String featureName = labelFeature.substring(labelFeature.indexOf("-") + 1);
				double w = Double.valueOf(featureParameters.get("w"));
				double G = Double.valueOf(featureParameters.get("G"));
				int labelIndex = Integer.valueOf(featureParameters.get("labelIndex"));
				int featureIndex = Integer.valueOf(featureParameters.get("featureIndex"));
				
				int index = labelIndex*this.numFeatures+featureIndex;
				this.featureNames.put(featureIndex, featureName);
				this.feature_w.put(index, w);
				this.feature_G.put(index, G);
			} else if (assignmentLeft.equals("labelBias")) {
				SerializationUtil.deserializeGenericName(reader);
				Map<String, String> biasParameters = SerializationUtil.deserializeArguments(reader);
				double b = Double.valueOf(biasParameters.get("b"));
				double G = Double.valueOf(biasParameters.get("G"));
				int index = Integer.valueOf(biasParameters.get("index"));
				
				this.bias_b[index] = b;
				this.bias_G[index] = G;
			} else {
				break;
			}
		}
		
		return true;
	}
	
	@Override
	protected boolean serializeParameters(Writer writer) throws IOException {
		Pair<String, String> tAssignment = new Pair<String, String>("t", String.valueOf(this.t));
		if (!SerializationUtil.serializeAssignment(tAssignment, writer))
			return false;
		writer.write("\n");
		
		Pair<String, String> numFeatureWeightsAssignment = new Pair<String, String>("numWeights", String.valueOf(this.labelIndices.size()*this.numFeatures));
		if (!SerializationUtil.serializeAssignment(numFeatureWeightsAssignment, writer))
			return false;
		writer.write("\n"); 
		
		if (this.labelIndices.size() == 2)
			return serializeParametersBinary(writer);
		
		for (int i = 0; i < this.labelIndices.size(); i++) {
			String label = this.labelIndices.reverseGet(i).toString();
			String biasValue = label +
					  "(b=" + this.bias_b[i] +
					  ", G=" + this.bias_G[i] +
					  ", index=" + i +
					  ")";

			Pair<String, String> biasAssignment = new Pair<String, String>("labelBias", biasValue);
			if (!SerializationUtil.serializeAssignment(biasAssignment, writer))
				return false;
			writer.write("\n");
		}
		
		List<Entry<Integer, Double>> wList = new ArrayList<Entry<Integer, Double>>(this.feature_w.entrySet());
		Collections.sort(wList, new Comparator<Entry<Integer, Double>>() {
			@Override
			public int compare(Entry<Integer, Double> e1,
					Entry<Integer, Double> e2) {
				if (Math.abs(e1.getValue()) > Math.abs(e2.getValue()))
					return -1;
				else if (Math.abs(e1.getValue()) < Math.abs(e2.getValue()))
					return 1;
				else
					return 0;
			} });
		
		for (Entry<Integer, Double> weightEntry : wList) {
			int weightIndex = weightEntry.getKey();
			int labelIndex = getLabelIndex(weightIndex);
			int featureIndex = getFeatureIndex(weightIndex);
			String label = this.labelIndices.reverseGet(labelIndex).toString();
			String featureName = this.featureNames.get(featureIndex);
			double w = weightEntry.getValue();
			double G = (this.feature_G.containsKey(weightIndex)) ? this.feature_G.get(weightIndex) : 0;
			
			if (w == 0)
				continue;
			
			String featureValue = label + "-" + 
					  featureName + 
					  "(w=" + w +
					  ", G=" + G +
					  ", labelIndex=" + labelIndex +
					  ", featureIndex=" + featureIndex + 
					  ")";

			Pair<String, String> featureAssignment = new Pair<String, String>("labelFeature", featureValue);
			if (!SerializationUtil.serializeAssignment(featureAssignment, writer))
				return false;
			writer.write("\n");
		}

		writer.write("\n");
		
		return true;
	}
	
	protected boolean serializeParametersBinary(Writer writer) throws IOException {
		int onIndex = 0;
		int offIndex = 1;
		
		if (this.labelIndices.reverseGet(0).toString().equals("true")) {
			onIndex = 0;
			offIndex = 1;
		} else {
			onIndex = 1;
			offIndex = 0;
		}
		
		double b = this.bias_b[onIndex] - this.bias_b[offIndex];
		String biasValue = "true_false(b=" + b + ")";
	
		Pair<String, String> biasAssignment = new Pair<String, String>("labelBias", biasValue);
		if (!SerializationUtil.serializeAssignment(biasAssignment, writer))
			return false;
		writer.write("\n");
		
		Map<Integer, Double> mergedWeights = new HashMap<Integer, Double>();
		for (Entry<Integer, Double> weightEntry : this.feature_w.entrySet()) {
			int labelIndex = getLabelIndex(weightEntry.getKey());
			int featureIndex = getFeatureIndex(weightEntry.getKey());
			double delta = weightEntry.getValue();
			if (labelIndex != onIndex)
				delta = -delta;
			if (!mergedWeights.containsKey(featureIndex))
				mergedWeights.put(featureIndex, 0.0);
			mergedWeights.put(featureIndex, mergedWeights.get(featureIndex) + delta);
		}
		
		List<Entry<Integer, Double>> wList = new ArrayList<Entry<Integer, Double>>(mergedWeights.entrySet());
		Collections.sort(wList, new Comparator<Entry<Integer, Double>>() {
			@Override
			public int compare(Entry<Integer, Double> e1,
					Entry<Integer, Double> e2) {
				if (e1.getValue() > e2.getValue())
					return -1;
				else if (e1.getValue() < e2.getKey())
					return 1;
				else
					return 0;
			} });
		
		for (Entry<Integer, Double> weightEntry : wList) {
			int featureIndex = weightEntry.getKey();
			String label = "true_false";
			String featureName = this.featureNames.get(featureIndex);
			double w = weightEntry.getValue();
			
			if (w == 0)
				continue;
			
			String featureValue = label + "-" + 
					  featureName + 
					  "(w=" + w +
					  ", featureIndex=" + featureIndex + 
					  ")";

			Pair<String, String> featureAssignment = new Pair<String, String>("labelFeature", featureValue);
			if (!SerializationUtil.serializeAssignment(featureAssignment, writer))
				return false;
			writer.write("\n");
		}

		writer.write("\n");
		
		return true;
	}

	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelSVM<D, L>();
	}

	@Override
	public String getGenericName() {
		return "SVM";
	}

	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> posteriors = new HashMap<D, Map<L, Double>>(data.size());

		for (D datum : data) {
			posteriors.put(datum, posteriorForDatum(data, datum));
		}
		
		return posteriors;
	}

	/**
	 * 
	 * @param data
	 * @param datum
	 * @return posterior based on softmax using scores for labels assigned to datum
	 */
	protected Map<L, Double> posteriorForDatum(FeaturizedDataSet<D, L> data, D datum) {
		Map<L, Double> posterior = new HashMap<L, Double>(this.validLabels.size());
		double[] scores = new double[this.validLabels.size()];
		double max = Double.NEGATIVE_INFINITY;
		for (L label : this.validLabels) {
			double score = scoreLabel(data, datum, label, false);
			scores[this.labelIndices.get(label)] = score;
			if (score > max)
				max = score;
		}
		
		double lse = 0;
		for (int i = 0; i < scores.length; i++)
			lse += Math.exp(scores[i] - max);
		lse = max + Math.log(lse);
		
		for (L label : this.validLabels) {
			posterior.put(label, Math.exp(scores[this.labelIndices.get(label)]-lse));
		}
		
		return posterior;
	}
	
	@Override
	public Map<D, L> classify(FeaturizedDataSet<D, L> data) {
		Map<D, L> classifiedData = new HashMap<D, L>();
		
		for (D datum : data) {
			classifiedData.put(datum, argMaxScoreLabel(data, datum, false));
		}
	
		return classifiedData;
	}
}
