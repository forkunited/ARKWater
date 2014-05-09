package ark.model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.annotation.structure.DatumStructure;
import ark.data.annotation.structure.DatumStructureCollection;
import ark.data.feature.FeaturizedDataSet;
import ark.model.cost.FactoredCost;
import ark.util.BidirectionalLookupTable;
import ark.util.OutputWriter;
import ark.util.Pair;
import ark.util.SerializationUtil;

public class SupervisedModelStructuredSVMC<D extends Datum<L>, L> extends SupervisedModel<D, L> {
	private BidirectionalLookupTable<L, Integer> labelIndices;
	private FactoredCost<D, L> factoredCost;
	private String datumStructureOptimizer;
	private String datumStructureCollection;
	
	private int trainingIterations;
	private int t;
	private Map<Integer, String> featureNames;
	private double[] feature_w; // Labels x Input features
	private double[] feature_u; 
	private double[] feature_G;  // Just diagonal
	private double[] bias_b;
	private double[] bias_u;
	private double[] bias_G;
	private double[] cost_v;
	private double[] cost_u; 
	private double[] cost_G;  // Just diagonal
	private Integer[] cost_i; // Cost indices for sorting v and G
	
	private double l1;
	private double l2;
	private double n = 1.0;
	private double epsilon = 0;
	private String[] hyperParameterNames = { "l2", "l1", "c", "n", "epsilon" };
	
	private class CostWeightComparator implements Comparator<Integer> {
	    @Override
	    public int compare(Integer i1, Integer i2) {
	    	double u_1 = cost_G[i1]*(2.0*cost_v[i1]-1);
	    	double u_2 = cost_G[i2]*(2.0*cost_v[i2]-1);
	    	
	    	if (cost_G[i1] != 0 && cost_G[i2] == 0)
	    		return -1;
	    	else if (cost_G[i1] == 0 && cost_G[i2] != 0)
	    		return 1;
	    	if (u_1 > u_2)
	    		return -1;
	    	else if (u_1 < u_2)
	    		return 1;
	    	else 
	    		return 0;
	    }
	}
	
	public SupervisedModelStructuredSVMC() {
		this.featureNames = new HashMap<Integer, String>();
	}
	
	@Override
	protected boolean deserializeExtraInfo(String name, BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		if (this.validLabels != null && this.labelIndices == null) {
			// Might be better to do this somewhere else...?
			this.labelIndices = new BidirectionalLookupTable<L, Integer>();
			int i = 0;
			for (L label : this.validLabels) {
				this.labelIndices.put(label, i);
				i++;
			}
		}
		
		if (name.equals("trainingIterations")) {
			this.trainingIterations = Integer.valueOf(SerializationUtil.deserializeAssignmentRight(reader));
		} else if (name.equals("factoredCost")) {
			String genericCost = SerializationUtil.deserializeGenericName(reader);
			this.factoredCost = datumTools.makeFactoredCostInstance(genericCost);
			if (!this.factoredCost.deserialize(reader, false, datumTools))
				return false;
		} else if (name.equals("datumStructureCollection")) {
			this.datumStructureCollection = SerializationUtil.deserializeAssignmentRight(reader);
		} else if (name.equals("datumStructureOptimizer")) {
			this.datumStructureOptimizer = SerializationUtil.deserializeAssignmentRight(reader);
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
		
		if (this.factoredCost != null) {
			writer.write("\t");
			Pair<String, String> factoredCostAssignment = new Pair<String, String>("factoredCost", this.factoredCost.toString(false));
			if (!SerializationUtil.serializeAssignment(factoredCostAssignment, writer))
				return false;
			writer.write("\n");
		}
		
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
		
		return true;
	}

	@Override
	public boolean train(FeaturizedDataSet<D, L> data) {
		OutputWriter output = data.getDatumTools().getDataTools().getOutputWriter();
		
		if (!this.factoredCost.init(this, data))
			return false;
		
		if (this.feature_w == null) {
			this.t = 1;
			this.feature_w = new double[data.getFeatureVocabularySize()*this.validLabels.size()];
			this.feature_u = new double[this.feature_w.length];
			this.feature_G = new double[this.feature_w.length];
			
			this.bias_b = new double[this.validLabels.size()];
			this.bias_u = new double[this.bias_b.length];
			this.bias_G = new double[this.bias_u.length];
			
			this.cost_v = new double[this.factoredCost.getVocabularySize()];
			this.cost_u = new double[this.cost_v.length];
			this.cost_G = new double[this.cost_v.length];
		
			this.cost_i = new Integer[this.cost_v.length];
			for (int i = 0; i < this.cost_i.length; i++)
				this.cost_i[i] = i;
		}
		
		DatumStructureCollection<D, L> datumStructureCollection = data.getDatumTools().makeDatumStructureCollection(this.datumStructureCollection, data);
		
		output.debugWriteln("Training StructuredSVMC for " + this.trainingIterations + " iterations on structures " + datumStructureCollection.size() + " in " + this.datumStructureCollection + "...");

		double prevObjectiveValue = objectiveValue(data, datumStructureCollection);
		Map<D, L> prevPredictions = makeQuickPredictions(data);
		
		CostWeightComparator costWeightComparator = new CostWeightComparator();
		double[] feature_g = new double[this.feature_w.length];
		double[] bias_g = new double[this.bias_b.length];
		double[] cost_g = new double[this.cost_v.length];
		
		for (int iteration = 0; iteration < this.trainingIterations; iteration++) {
			for (DatumStructure<D, L> datumStructure : datumStructureCollection) {
				Map<D, Map<L, Double>> scoredDatumLabels = scoreDatumStructureLabels(data, datumStructure, true);
				Map<D, L> datumLabels = datumStructure.getDatumLabels(this.labelMapping);
				// Maybe just optimize here...?
				Map<D, L> bestDatumLabels = getBestDatumLabels(data, datumStructure, scoredDatumLabels);

				Map<Integer, Double> datumStructureFeatureValues = computeDatumStructureFeatureValues(data, datumStructure, datumLabels, iteration == 0);
				Map<Integer, Double> bestStructureFeatureValues = computeDatumStructureFeatureValues(data, datumStructure, bestDatumLabels, false);
				Map<Integer, Double> bestStructureCosts = computeDatumStructureCosts(datumStructure, bestDatumLabels);
				
				// Update feature weights
				for (int i = 0; i < this.feature_w.length; i++) { 
					double datumFeatureValue = (datumStructureFeatureValues.containsKey(i)) ? datumStructureFeatureValues.get(i) : 0.0;
					double bestFeatureValue = (bestStructureFeatureValues.containsKey(i)) ? bestStructureFeatureValues.get(i) : 0.0;
					
					if (this.l1 == 0 && this.feature_w[i] == 0 && datumFeatureValue == bestFeatureValue)
						continue;
					
					feature_g[i] = this.l2*this.feature_w[i]-datumFeatureValue+bestFeatureValue;
					
					this.feature_G[i] += feature_g[i]*feature_g[i];
					this.feature_u[i] += feature_g[i];
					
					if (this.feature_G[i] == 0)
						continue;
					if (this.l1 == 0)
						this.feature_w[i] -= feature_g[i]*this.n/Math.sqrt(this.feature_G[i]); 
					else {
						if (Math.abs(this.feature_u[i])/this.t <= this.l1)
							this.feature_w[i] = 0; 
						else 
							this.feature_w[i] = -Math.signum(this.feature_u[i])*(this.t*this.n/(Math.sqrt(this.feature_G[i])))*((Math.abs(this.feature_u[i])/this.t)-this.l1); 
					}
				}
				
				// Update label biases
				for (int i = 0; i < this.bias_b.length; i++) {
					L label = this.labelIndices.reverseGet(i);
					int datumLabelCount = getLabelCount(datumLabels, label);
					int bestLabelCount = getLabelCount(bestDatumLabels, label);
					bias_g[i] = -datumLabelCount + bestLabelCount;
					
					this.bias_G[i] += bias_g[i]*bias_g[i];
					this.bias_u[i] += bias_g[i];
					if (this.bias_G[i] == 0)
						continue;
					this.bias_b[i] -= bias_g[i]*this.n/Math.sqrt(this.bias_G[i]);
				}
				
				// Update cost weights
				for (int i = 0; i < this.cost_v.length; i++) {
					cost_g[i] = (bestStructureCosts.containsKey(i)) ? bestStructureCosts.get(i) : 0;
					this.cost_G[i] += cost_g[i]*cost_g[i];
					this.cost_u[i] += cost_g[i];
					
					if (this.cost_G[i] != 0)
						this.cost_v[i] -= cost_g[i]*this.n/Math.sqrt(this.cost_G[i]); 
				}
				
				// Project cost weights onto simplex \sum v_i = 1, v_i >= 0
				// Find p = max { j : u_j - 1/G_j((\sum^j u_i) - 1.0)/(\sum^j 1.0/G_i) > 0 } 
				// where u and G are sorted desc
				Arrays.sort(this.cost_i, costWeightComparator);
				double sumV = 0;
				double harmonicG = 0;
				double theta = 0;
				for (int p = 0; p < this.cost_v.length; p++) {
					if (this.cost_G[this.cost_i[p]] != 0) {
						sumV += this.cost_v[this.cost_i[p]];
						harmonicG += 1.0/this.cost_G[this.cost_i[p]];
					}
					double prevTheta = theta;
					theta = (sumV-1.0)/harmonicG;
					if (this.cost_G[this.cost_i[p]] == 0 || this.cost_v[this.cost_i[p]]-theta/this.cost_G[this.cost_i[p]] <= 0) {
						theta = prevTheta;
						break;
					}
				}
				
				for (int j = 0; j < this.cost_v.length; j++) {
					if (this.cost_G[j] == 0)
						this.cost_v[j] = 0;
					else
						this.cost_v[j] = Math.max(0, this.cost_v[j]-theta/this.cost_G[j]);
				}
				
				this.t++;
			}
			
			double objectiveValue = objectiveValue(data, datumStructureCollection);
			double objectiveValueDiff = objectiveValue - prevObjectiveValue;
			Map<D, L> predictions = makeQuickPredictions(data);
			int labelDifferences = countLabelDifferences(prevPredictions, predictions);
			
			double vSum = 0;
			for (int i = 0; i < this.cost_v.length; i++)
				vSum += this.cost_v[i];
			
			output.debugWriteln("(c=" + this.factoredCost.getParameterValue("c")  + ", l1=" + this.l1 + ", l2=" + this.l2 + ") Finished iteration " + iteration + " objective diff: " + objectiveValueDiff + " objective: " + objectiveValue + " prediction-diff: " + labelDifferences + "/" + predictions.size() + " v-sum: " + vSum + ").");
			
			if (iteration > 20 && Math.abs(objectiveValueDiff) < this.epsilon) {
				output.debugWriteln("(c=" + this.factoredCost.getParameterValue("c")  + ", l1=" + this.l1 + ", l2=" + this.l2 + ") Terminating early at iteration " + iteration);
				break;
			}
			
			prevObjectiveValue = objectiveValue;
			prevPredictions = predictions;
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
	
	private double objectiveValue(FeaturizedDataSet<D, L> data, DatumStructureCollection<D, L> datumStructureCollection) {
		double value = 0;
		
		if (this.l1 > 0) {
			double l1Norm = 0;
			for (int i = 0; i < this.feature_w.length; i++)
				value += Math.abs(this.feature_w[i]);
			value += l1Norm*this.l1;
		}
		
		if (this.l2 > 0) {
			double l2Norm = 0;
			for (int i = 0; i < this.feature_w.length; i++)
				value += this.feature_w[i]*this.feature_w[i];
			value += l2Norm*this.l2*.5;
		}
		
		for (DatumStructure<D, L> datumStructure : datumStructureCollection) {
			Map<D, Map<L, Double>> scoredDatumLabels = scoreDatumStructureLabels(data, datumStructure, true);
			Map<D, L> datumLabels = datumStructure.getDatumLabels(this.labelMapping);
			Map<D, L> bestDatumLabels = getBestDatumLabels(data, datumStructure, scoredDatumLabels);
		
			double datumStructureScore = scoreDatumStructure(data, datumStructure, datumLabels, false);
			double bestStructureScore = scoreDatumStructure(data, datumStructure, bestDatumLabels, true);
		
			value += bestStructureScore - datumStructureScore;
		}
		
		return value;
	}
	
	private int getLabelCount(Map<D, L> datumsToLabels, L countLabel) {
		int count = 0;
		for (L label : datumsToLabels.values())
			if (label.equals(countLabel))
				count++;
		return count;
	}
	
	private Map<D, Map<L, Double>> scoreDatumStructureLabels(FeaturizedDataSet<D, L> data, DatumStructure<D, L> datumStructure, boolean includeCost) {
		Map<D, Map<L, Double>> datumLabelScores = new HashMap<D, Map<L, Double>>();
		
		for (D datum : datumStructure) {
			datumLabelScores.put(datum, new HashMap<L, Double>());
			for (L label : this.validLabels) {
				datumLabelScores.get(datum).put(label,
						scoreDatumLabel(data, datum, label, includeCost));
			}
		}
		
		return datumLabelScores;
	}
	
	private Map<Integer, Double> computeDatumStructureCosts(DatumStructure<D, L> datumStructure, Map<D, L> labels) {
		Map<Integer, Double> costs = new HashMap<Integer, Double>();
		
		for (D datum : datumStructure) {
			Map<Integer, Double> datumCosts = this.factoredCost.computeVector(datum, labels.get(datum));
			for (Entry<Integer, Double> entry : datumCosts.entrySet()) {
				if (!costs.containsKey(entry.getKey()))
					costs.put(entry.getKey(), 0.0);
				costs.put(entry.getKey(), costs.get(entry.getKey()) + entry.getValue());
			}
		}
		
		return costs;
	}
	
	private double scoreDatumStructure(FeaturizedDataSet<D, L> data, DatumStructure<D, L> datumStructure, Map<D, L> structureLabels, boolean includeCost) {
		double score = 0.0;
		
		Map<Integer, Double> datumStructureFeatureValues = computeDatumStructureFeatureValues(data, datumStructure, structureLabels, false);
		for (Entry<Integer, Double> entry : datumStructureFeatureValues.entrySet()) {
			score += this.feature_w[entry.getKey()]*entry.getValue();
		}
		
		for (int i = 0; i < this.bias_b.length; i++) {
			L label = this.labelIndices.reverseGet(i);
			int datumLabelCount = getLabelCount(structureLabels, label);
			score += this.bias_b[i]*datumLabelCount;
		}
		
		if (includeCost) {
			Map<Integer, Double> datumStructureCosts = computeDatumStructureCosts(datumStructure, structureLabels);
			for (Entry<Integer, Double> entry : datumStructureCosts.entrySet()) {
				score += this.cost_v[entry.getKey()]*entry.getValue();
			}
		}
		
		return score;
	}
	
	private double scoreDatumLabel(FeaturizedDataSet<D, L> data, D datum, L label, boolean includeCost) {
		double score = 0;		

		Map<Integer, Double> featureValues = data.getFeatureVocabularyValues(datum);
		int labelIndex = this.labelIndices.get(label);
		int numFeatures = data.getFeatureVocabularySize();
		int weightIndexOffset = labelIndex*numFeatures;
		for (Entry<Integer, Double> entry : featureValues.entrySet()) {
			score += this.feature_w[weightIndexOffset + entry.getKey()]*entry.getValue();
		}
		
		score += this.bias_b[labelIndex];

		if (includeCost) {
			Map<Integer, Double> costs = this.factoredCost.computeVector(datum, label);
			for (Entry<Integer, Double> entry : costs.entrySet())
				score += costs.get(entry.getKey())*this.cost_v[entry.getKey()];
		}
		
		return score;
	}

	private Map<Integer, Double> computeDatumStructureFeatureValues(FeaturizedDataSet<D,L> data, DatumStructure<D, L> datumStructure, Map<D, L> structureLabels, boolean cacheFeatureNames) {
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

	private Map<D, L> getBestDatumLabels(FeaturizedDataSet<D, L> data, DatumStructure<D, L> datumStructure, Map<D, Map<L, Double>> scoredDatumLabels) {
		Map<D, L> optimizedDatumLabels = datumStructure.optimize(this.datumStructureOptimizer, scoredDatumLabels, this.fixedDatumLabels, this.validLabels, this.labelMapping);
		Map<D, L> actualDatumLabels = datumStructure.getDatumLabels(this.labelMapping);
		
		double optimizedScore = scoreDatumStructure(data, datumStructure, optimizedDatumLabels, true);
		double actualScore = scoreDatumStructure(data, datumStructure, actualDatumLabels, false);
		
		if (actualScore > optimizedScore)
			return actualDatumLabels;
		else
			return optimizedDatumLabels;
	}
	
	private Map<D, L> makeQuickPredictions(FeaturizedDataSet<D, L> data) {
		Map<D, L> predictions = new HashMap<D, L>();
		DatumStructureCollection<D, L> datumStructureCollection = data.getDatumTools().makeDatumStructureCollection(this.datumStructureCollection, data);
		
		for (DatumStructure<D, L> datumStructure : datumStructureCollection) {
			Map<D, Map<L, Double>> scoredDatumLabels = scoreDatumStructureLabels(data, datumStructure, false);
			predictions.putAll(
				datumStructure.optimize(this.datumStructureOptimizer, scoredDatumLabels, this.fixedDatumLabels, this.validLabels, this.labelMapping)
			);
		}
		
		return predictions;
	}
	
	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> posteriors = new HashMap<D, Map<L, Double>>(data.size());
		DatumStructureCollection<D, L> datumStructureCollection = data.getDatumTools().makeDatumStructureCollection(this.datumStructureCollection, data);
		Map<D, L> bestDatumLabels = new HashMap<D, L>();
		
		if (this.factoredCost != null && !this.factoredCost.init(this, data))
			return null;

		
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
	
	@Override
	protected String[] getHyperParameterNames() {
		return this.hyperParameterNames;
	}

	@Override
	public String getHyperParameterValue(String parameter) {
		if (parameter.equals("l1"))
			return String.valueOf(this.l1);
		else if (parameter.equals("l2"))
			return String.valueOf(this.l2);
		else if (parameter.equals("c"))
			return (this.factoredCost == null) ? "0" : this.factoredCost.getParameterValue("c");
		else if (parameter.equals("n"))
			return String.valueOf(this.n);
		else if (parameter.equals("epsilon"))
			return String.valueOf(this.epsilon);
		return null;
	}

	@Override
	public boolean setHyperParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("l1"))
			this.l1 = Double.valueOf(parameterValue);
		else if (parameter.equals("l2"))
			this.l2 = Double.valueOf(parameterValue);
		else if (parameter.equals("c") && this.factoredCost != null)
			this.factoredCost.setParameterValue("c", parameterValue, datumTools);
		else if (parameter.equals("n"))
			this.n = Double.valueOf(parameterValue);
		else if (parameter.equals("epsilon"))
			this.epsilon = Double.valueOf(parameterValue);
		else
			return false;
		return true;
	}
	
	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelStructuredSVMC<D, L>();
	}
	
	@Override
	public String getGenericName() {
		return "StructuredSVMC";
	}
	
	@Override
	protected boolean deserializeParameters(BufferedReader reader,
			Tools<D, L> datumTools) throws IOException {
		Pair<String, String> tAssign = SerializationUtil.deserializeAssignment(reader);
		Pair<String, String> numWeightsAssign = SerializationUtil.deserializeAssignment(reader);
		Pair<String, String> numCostsAssign = SerializationUtil.deserializeAssignment(reader);
		
		int numWeights = Integer.valueOf(numWeightsAssign.getSecond());
		int numCosts = Integer.valueOf(numCostsAssign.getSecond());
		int numFeatures = numWeights / this.labelIndices.size();
		
		this.t = Integer.valueOf(tAssign.getSecond());
		this.featureNames = new HashMap<Integer, String>();
		
		this.feature_w = new double[numWeights];
		this.feature_u = new double[this.feature_w.length];
		this.feature_G = new double[this.feature_w.length];
		
		this.bias_b = new double[this.labelIndices.size()];
		this.bias_u = new double[this.bias_b.length];
		this.bias_G = new double[this.bias_b.length];
		
		this.cost_v = new double[numCosts];
		this.cost_u = new double[this.cost_v.length];
		this.cost_G = new double[this.cost_v.length];
	
		this.cost_i = new Integer[this.cost_v.length];
		for (int i = 0; i < this.cost_i.length; i++)
			this.cost_i[i] = i;
		
		String assignmentLeft = null;
		while ((assignmentLeft = SerializationUtil.deserializeAssignmentLeft(reader)) != null) {
			if (assignmentLeft.equals("labelFeature")) {
				String labelFeature = SerializationUtil.deserializeGenericName(reader);
				Map<String, String> featureParameters = SerializationUtil.deserializeArguments(reader);
				
				String featureName = labelFeature.substring(labelFeature.indexOf("-") + 1);
				double w = Double.valueOf(featureParameters.get("w"));
				double G = Double.valueOf(featureParameters.get("G"));
				double u = Double.valueOf(featureParameters.get("u"));
				int labelIndex = Integer.valueOf(featureParameters.get("labelIndex"));
				int featureIndex = Integer.valueOf(featureParameters.get("featureIndex"));
				
				int index = labelIndex*numFeatures+featureIndex;
				this.featureNames.put(featureIndex, featureName);
				this.feature_w[index] = w;
				this.feature_u[index] = u;
				this.feature_G[index] = G;
			} else if (assignmentLeft.equals("labelBias")) {
				SerializationUtil.deserializeGenericName(reader);
				Map<String, String> biasParameters = SerializationUtil.deserializeArguments(reader);
				double b = Double.valueOf(biasParameters.get("b"));
				double G = Double.valueOf(biasParameters.get("G"));
				double u = Double.valueOf(biasParameters.get("u"));
				int index = Integer.valueOf(biasParameters.get("index"));
				
				this.bias_b[index] = b;
				this.bias_G[index] = G;
				this.bias_u[index] = u;
			} else if (assignmentLeft.equals("cost")) {
				SerializationUtil.deserializeGenericName(reader);
				Map<String, String> costParameters = SerializationUtil.deserializeArguments(reader);
				double v = Double.valueOf(costParameters.get("v"));
				double G = Double.valueOf(costParameters.get("G"));
				double u = Double.valueOf(costParameters.get("u"));
				int index = Integer.valueOf(costParameters.get("index"));
				
				this.cost_v[index] = v;
				this.cost_G[index] = G;
				this.cost_u[index] = u;
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
		
		Pair<String, String> numFeatureWeightsAssignment = new Pair<String, String>("numWeights", String.valueOf(this.feature_w.length));
		if (!SerializationUtil.serializeAssignment(numFeatureWeightsAssignment, writer))
			return false;
		writer.write("\n");
		
		Pair<String, String> numCostWeightsAssignment = new Pair<String, String>("numCosts", String.valueOf(this.cost_v.length));
		if (!SerializationUtil.serializeAssignment(numCostWeightsAssignment, writer))
			return false;
		writer.write("\n");
		
		for (int i = 0; i < this.labelIndices.size(); i++) {
			String label = this.labelIndices.reverseGet(i).toString();
			String biasValue = label +
					  "(b=" + this.bias_b[i] +
					  ", G=" + this.bias_G[i] +
					  ", u=" + this.bias_u[i] +
					  ", index=" + i +
					  ")";

			Pair<String, String> biasAssignment = new Pair<String, String>("labelBias", biasValue);
			if (!SerializationUtil.serializeAssignment(biasAssignment, writer))
				return false;
			writer.write("\n");
		}
		
		for (int i = 0; i < this.labelIndices.size(); i++) {
			String label = this.labelIndices.reverseGet(i).toString();
			for (Entry<Integer, String> featureName : this.featureNames.entrySet()) {
				int index = i*this.feature_w.length/this.labelIndices.size()+featureName.getKey();
				
				String featureValue = label + "-" + 
									  featureName.getValue() + 
									  "(w=" + this.feature_w[index] +
									  ", G=" + this.feature_G[index] +
									  ", u=" + this.feature_u[index] +
									  ", labelIndex=" + i +
									  ", featureIndex=" + featureName.getKey() + 
									  ")";
				
				Pair<String, String> featureAssignment = new Pair<String, String>("labelFeature", featureValue);
				if (!SerializationUtil.serializeAssignment(featureAssignment, writer))
					return false;
				writer.write("\n");
			}
		}
		
		if (this.factoredCost != null) {
			List<String> costNames = this.factoredCost.getSpecificShortNames();
			for (int i = 0; i < costNames.size(); i++) {
				String costValue = costNames.get(i) +
								   "(v=" + this.cost_v[i] +
								   ", G=" + this.cost_G[i] +
								   ", u=" + this.cost_u[i] +
								   ", index=" + i +
								   ")";
				
				Pair<String, String> costAssignment = new Pair<String, String>("cost", costValue);
				if (!SerializationUtil.serializeAssignment(costAssignment, writer))
					return false;
				writer.write("\n");
			}
		}

		writer.write("\n");
		
		return true;
	}
	
	public SupervisedModel<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		SupervisedModelStructuredSVMC<D, L> clone = (SupervisedModelStructuredSVMC<D, L>)super.clone(datumTools, environment);
		
		clone.labelIndices = this.labelIndices;
		clone.trainingIterations = this.trainingIterations;
		clone.datumStructureCollection = this.datumStructureCollection;
		clone.datumStructureOptimizer = this.datumStructureOptimizer;
		if (this.factoredCost != null) {
			clone.factoredCost = this.factoredCost.clone(datumTools, environment);
		}
		
		return clone;
	}
}
