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
import ark.data.feature.FeaturizedDataSet;
import ark.model.cost.FactoredCost;
import ark.util.BidirectionalLookupTable;
import ark.util.OutputWriter;
import ark.util.Pair;
import ark.util.SerializationUtil;

public class SupervisedModelSVMCAlt<D extends Datum<L>, L> extends SupervisedModel<D, L> {
	private BidirectionalLookupTable<L, Integer> labelIndices;
	private FactoredCost<D, L> factoredCost;
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
	private Integer[] cost_i; // Cost indices for sorting v and G
	
	private double l1;
	private double l2;
	private double n = 1.0;
	private double epsilon = 0;
	private String[] hyperParameterNames = { "l2", "l1", "c", "n", "epsilon" };
	
	private class CostWeightComparator implements Comparator<Integer> {
	    @Override
	    public int compare(Integer i1, Integer i2) {
	    	double u_1 = cost_v[i1];
	    	double u_2 = cost_v[i2];
	    	
	    	if (u_1 > u_2)
	    		return -1;
	    	else if (u_1 < u_2)
	    		return 1;
	    	else 
	    		return 0;
	    }
	}
	
	public SupervisedModelSVMCAlt() {
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
		
			this.cost_i = new Integer[this.cost_v.length];
			for (int i = 0; i < this.cost_i.length; i++)
				this.cost_i[i] = i;
		}
		
		double prevObjectiveValue = objectiveValue(data);
		Map<D, L> prevPredictions = makeQuickPredictions(data);
		
		output.debugWriteln("Training SVMCAlt for " + this.trainingIterations + " iterations...");
		
		double[] feature_g = new double[this.feature_w.length];
		double[] bias_g = new double[this.bias_b.length];
		
		for (int iteration = 0; iteration < this.trainingIterations; iteration++) {
			for (D datum : data) {	
				L datumLabel = this.mapValidLabel(datum.getLabel());
				L bestLabel = argMaxScoreLabel(data, datum, true);
				boolean datumLabelBest = datumLabel.equals(bestLabel);
				
				Map<Integer, Double> datumFeatureValues = data.getFeatureVocabularyValues(datum);
				
				if (iteration == 0) {
					List<Integer> missingNameKeys = new ArrayList<Integer>();
					for (Integer key : datumFeatureValues.keySet())
						if (!this.featureNames.containsKey(key))
							missingNameKeys.add(key);
					this.featureNames.putAll(data.getFeatureVocabularyNamesForIndices(missingNameKeys));
				}
				
				// Update feature weights
				for (int i = 0; i < this.feature_w.length; i++) {
					if (this.l1 == 0 && this.feature_w[i] == 0 && datumLabelBest)
						continue;
					
					feature_g[i] = this.l2*this.feature_w[i]-labelFeatureValue(data, datumFeatureValues, i, datumLabel)+labelFeatureValue(data, datumFeatureValues, i, bestLabel);
					
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
							this.feature_w[i] = -Math.signum(this.feature_u[i])*this.n*(this.t/(Math.sqrt(this.feature_G[i])))*((Math.abs(this.feature_u[i])/this.t)-this.l1); 
					}
				}
				
				if (datumLabelBest) {
					this.t++;
					continue;
				}
				
				// Update label biases
				for (int i = 0; i < this.bias_b.length; i++) {
					bias_g[i] = ((this.labelIndices.get(datumLabel) == i) ? -1.0 : 0.0) +
									(this.labelIndices.get(bestLabel) == i ? 1.0 : 0.0);
					
					this.bias_G[i] += bias_g[i]*bias_g[i];
					this.bias_u[i] += bias_g[i];
					if (this.bias_G[i] == 0)
						continue;
					this.bias_b[i] -= bias_g[i]*this.n/Math.sqrt(this.bias_G[i]);
				}
				
				this.t++;
			}
	
			if (!trainCostWeights(data))
				return false;
			
			double objectiveValue = objectiveValue(data);
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
	
	private Map<D, L> makeQuickPredictions(FeaturizedDataSet<D, L> data) {
		Map<D, L> predictions = new HashMap<D, L>();
		
		for (D datum : data) {
			double max = Double.NEGATIVE_INFINITY;
			L maxLabel = null;
			for (L label : this.validLabels) {
				double score = scoreLabel(data, datum, label, false);
				if (score > max) {
					max = score;
					maxLabel = label;
				}
			}
			
			predictions.put(datum, maxLabel);
		}
		
		return predictions;
	}
	
	private int countLabelDifferences(Map<D, L> labels1, Map<D, L> labels2) {
		int count = 0;
		for (Entry<D, L> entry: labels1.entrySet()) {
			if (!labels2.containsKey(entry.getKey()) && !entry.getValue().equals(labels2.get(entry.getKey())))
				count++;
		}
		return count;
	}
	
	private boolean trainCostWeights(FeaturizedDataSet<D, L> data) {
		Map<D, L> predictions = new HashMap<D, L>();
		for (D datum : data) {
			predictions.put(datum, argMaxScoreLabel(data, datum, true));
		}
		
		Map<Integer, Double> kappas = this.factoredCost.computeKappas(predictions);
		CostWeightComparator costWeightComparator = new CostWeightComparator();
		for (int i = 0; i < this.cost_v.length; i++) {
			if (!kappas.containsKey(i))
				this.cost_v[i] = 0;
			else
				this.cost_v[i] = -kappas.get(i);
		}
		
		// Project cost weights onto simplex \sum v_i = 1, v_i >= 0
		// Find p = max { j : u_j - (1/j)((\sum^j u_i) - 1.0) > 0 } 
		// where u is sorted desc
		Arrays.sort(this.cost_i, costWeightComparator);
		double sumV = 0;
		double theta = 0;
		for (int p = 0; p < this.cost_v.length; p++) {
			sumV += this.cost_v[this.cost_i[p]];
			double prevTheta = theta;
			theta = (sumV-1.0)/p;
			if (this.cost_v[this.cost_i[p]]-theta <= 0) {
				theta = prevTheta;
				break;
			}
		}
		
		for (int j = 0; j < this.cost_v.length; j++) {
			this.cost_v[j] = Math.max(0, this.cost_v[j]-theta);
		}
		
		return true;
	}
	
	private double objectiveValue(FeaturizedDataSet<D, L> data) {
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
		
		for (D datum : data) {
			double maxScore = maxScoreLabel(data, datum, true);
			double datumScore = scoreLabel(data, datum, datum.getLabel(), false);
			value += maxScore - datumScore;
		}
		
		return value;
	}
	
	private double maxScoreLabel(FeaturizedDataSet<D, L> data, D datum, boolean includeCost) {
		double maxScore = Double.NEGATIVE_INFINITY;
		for (L label : this.validLabels) {
			double score = scoreLabel(data, datum, label, includeCost);
			if (score >= maxScore) {
				maxScore = score;
			}
		}
		return maxScore;
	}
	
	private L argMaxScoreLabel(FeaturizedDataSet<D, L> data, D datum, boolean includeCost) {
		L maxLabel = null;
		double maxScore = Double.NEGATIVE_INFINITY;
		for (L label : this.validLabels) {
			double score = scoreLabel(data, datum, label, includeCost);
			if (score >= maxScore) {
				maxScore = score;
				maxLabel = label;
			}
		}
		return maxLabel;
	}
	
	private double scoreLabel(FeaturizedDataSet<D, L> data, D datum, L label, boolean includeCost) {
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
	
	private double labelFeatureValue(FeaturizedDataSet<D,L> data, Map<Integer, Double> featureValues, int weightIndex, L label) {
		int labelIndex = this.labelIndices.get(label);
		int numFeatures = data.getFeatureVocabularySize();
		int featureLabelIndex = weightIndex / numFeatures;
		if (featureLabelIndex != labelIndex)
			return 0.0;
		
		int featureIndex = weightIndex % numFeatures;
		if (!featureValues.containsKey(featureIndex))
			return 0.0;
		else
			return featureValues.get(featureIndex);
	}

	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> posteriors = new HashMap<D, Map<L, Double>>(data.size());
		if (this.factoredCost != null && !this.factoredCost.init(this, data))
			return null;
		for (D datum : data) {
			posteriors.put(datum, posteriorForDatum(data, datum));
		}
		
		return posteriors;
	}

	private Map<L, Double> posteriorForDatum(FeaturizedDataSet<D, L> data, D datum) {
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
		return new SupervisedModelSVMCAlt<D, L>();
	}
	
	@Override
	public String getGenericName() {
		return "SVMCAlt";
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
				int index = Integer.valueOf(costParameters.get("index"));
				
				this.cost_v[index] = v;
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
		SupervisedModelSVMCAlt<D, L> clone = (SupervisedModelSVMCAlt<D, L>)super.clone(datumTools, environment);
		
		clone.labelIndices = this.labelIndices;
		clone.trainingIterations = this.trainingIterations;
		if (this.factoredCost != null) {
			clone.factoredCost = this.factoredCost.clone(datumTools, environment);
		}
		
		return clone;
	}
}
