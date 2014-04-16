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

public class SupervisedModelSVMCostLearner<D extends Datum<L>, L> extends SupervisedModel<D, L> {
	private BidirectionalLookupTable<L, Integer> labelIndices;
	private FactoredCost<D, L> factoredCost;
	private int trainingIterations;
	private int t;
	private Map<Integer, String> featureNames;
	private double[] feature_w; // Labels x Input features
	private double[] feature_u; 
	private double[] feature_G;  // Just diagonal
	private double[] cost_v;
	private double[] cost_u; 
	private double[] cost_G;  // Just diagonal
	private Integer[] cost_i; // Cost indices for sorting v and G
	
	private double l1;
	private String[] hyperParameterNames = { "l1", "c" };
	
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
	
	public SupervisedModelSVMCostLearner() {
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
			
			this.cost_v = new double[this.factoredCost.getVocabularySize()];
			this.cost_u = new double[this.cost_v.length];
			this.cost_G = new double[this.cost_v.length];
		
			this.cost_i = new Integer[this.cost_v.length];
			for (int i = 0; i < this.cost_i.length; i++)
				this.cost_i[i] = i;
		}
		
		double[] prevFeature_w = Arrays.copyOf(this.feature_w, this.feature_w.length);
		double[] prevCost_v = Arrays.copyOf(this.cost_v, this.cost_v.length);
		
		output.debugWriteln("Training SVMCostLearner for " + this.trainingIterations + " iterations...");
		
		CostWeightComparator costWeightComparator = new CostWeightComparator();
		double[] feature_g = new double[this.feature_w.length];
		double[] cost_g = new double[this.cost_v.length];
		
		for (int iteration = 0; iteration < this.trainingIterations; iteration++) {
			for (D datum : data) {	
				L datumLabel = this.mapValidLabel(datum.getLabel());
				L bestLabel = argMaxScoreLabel(data, datum, true);
				Map<Integer, Double> bestLabelCosts = this.factoredCost.computeVector(datum, bestLabel);
				
				Map<Integer, Double> datumFeatureValues = data.getFeatureVocabularyValues(datum);
				List<Integer> missingNameKeys = new ArrayList<Integer>();
				for (Integer key : datumFeatureValues.keySet())
					if (!this.featureNames.containsKey(key))
						missingNameKeys.add(key);
				this.featureNames.putAll(data.getFeatureVocabularyNamesForIndices(missingNameKeys));
				
				// Update feature weights
				for (int i = 0; i < this.feature_w.length; i++) { 
					feature_g[i] = -labelFeatureValue(data, datumFeatureValues, i, datumLabel)+labelFeatureValue(data, datumFeatureValues, i, bestLabel);
					
					this.feature_G[i] += feature_g[i]*feature_g[i];
					this.feature_u[i] += feature_g[i];
					
					if (this.feature_G[i] == 0)
						continue;
					if (this.l1 == 0)
						this.feature_w[i] -= feature_g[i]/Math.sqrt(this.feature_G[i]); 
					else {
						if (Math.abs(this.feature_u[i])/this.t <= this.l1)
							this.feature_w[i] = 0; 
						else 
							this.feature_w[i] = -Math.signum(this.feature_u[i])*(this.t/(Math.sqrt(this.feature_G[i])))*((Math.abs(this.feature_u[i])/this.t)-this.l1); 
					}
				}
				
				// Update cost weights
				for (int i = 0; i < this.cost_v.length; i++) {
					cost_g[i] = (bestLabelCosts.containsKey(i)) ? bestLabelCosts.get(i) : 0;
					this.cost_G[i] += cost_g[i]*cost_g[i];
					this.cost_u[i] += cost_g[i];
					
					if (this.cost_G[i] != 0)
						this.cost_v[i] -= cost_g[i]/Math.sqrt(this.cost_G[i]); 
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
	
			double vDiff = averageDiff(this.cost_v, prevCost_v);
			double wDiff = averageDiff(this.feature_w, prevFeature_w);
			double vDiffMax = maxDiff(this.cost_v, prevCost_v);
			double wDiffMax = maxDiff(this.feature_w, prevFeature_w);
			
			double vSum = 0;
			for (int i = 0; i < this.cost_v.length; i++)
				vSum += this.cost_v[i];
			
			output.debugWriteln("Finished iteration " + iteration + " (v-diff (avg, max): (" + vDiff + ", " + vDiffMax + ") w-diff (avg, max): (" + wDiff + ", " + wDiffMax + ") v-sum: " + vSum + ").");
			prevCost_v = Arrays.copyOf(this.cost_v, prevCost_v.length);
			prevFeature_w = Arrays.copyOf(this.feature_w, prevFeature_w.length);
		}
		
		return true;
	}
	
	private double maxDiff(double[] v1, double[] v2) {
		double diffMax = 0.0;
		for (int i = 0; i < v1.length; i++)
			diffMax = Math.max(Math.abs(v2[i] - v1[i]), diffMax);
		return diffMax;		
	}
	
	private double averageDiff(double[] v1, double[] v2) {
		double diffSum = 0.0;
		for (int i = 0; i < v1.length; i++)
			diffSum += Math.abs(v2[i] - v1[i]);
		return diffSum/v1.length;
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
		if (!this.factoredCost.init(this, data))
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
		else if (parameter.equals("c"))
			return (this.factoredCost == null) ? "0" : this.factoredCost.getParameterValue("c");
		return null;
	}

	@Override
	public boolean setHyperParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("l1"))
			this.l1 = Double.valueOf(parameterValue);
		else if (parameter.equals("c") && this.factoredCost != null)
			this.factoredCost.setParameterValue("c", parameterValue, datumTools);
		else
			return false;
		return true;
	}
	
	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelSVMCostLearner<D, L>();
	}
	
	@Override
	public String getGenericName() {
		return "SVMCostLearner";
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
		SupervisedModelSVMCostLearner<D, L> clone = (SupervisedModelSVMCostLearner<D, L>)super.clone(datumTools, environment);
		
		clone.labelIndices = this.labelIndices;
		clone.trainingIterations = this.trainingIterations;
		if (this.factoredCost != null) {
			clone.factoredCost = this.factoredCost.clone(datumTools, environment);
		}
		
		return clone;
	}
}
