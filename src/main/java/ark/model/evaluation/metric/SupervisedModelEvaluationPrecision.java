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

package ark.model.evaluation.metric;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.Context;
import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.parse.Obj;
import ark.util.Pair;

/**
 * SupervisedModelEvaluationPrecision computes the precision
 * (http://en.wikipedia.org/wiki/Precision_and_recall)
 * for a supervised classification model on a data set.
 * 
 * The 'weighted' parameter indicates whether the measure for a particular
 * label should be weighted by the labels frequency within the data set.
 * 
 * The 'filterLabel' parameter limits the precision calculation to a single 
 * label
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public class SupervisedModelEvaluationPrecision<D extends Datum<L>, L> extends SupervisedModelEvaluation<D, L> {
	private boolean weighted;
	private String filterLabel; // Must be stored as string to work with ValidationGSTBinary
	private String[] parameterNames = { "weighted", "filterLabel" };
	
	public SupervisedModelEvaluationPrecision() {
		
	}

	public SupervisedModelEvaluationPrecision(Context<D, L> context) {
		this.context = context;
	}
	
	@Override
	protected double compute(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data, Map<D, L> predictions) {
		List<Pair<L, L>> actualAndPredicted = this.getMappedActualAndPredictedLabels(predictions);

		Map<L, Double> weights = new HashMap<L, Double>();
		Map<L, Double> tps = new HashMap<L, Double>();
		Map<L, Double> fps = new HashMap<L, Double>();
		
		for (L label : model.getValidLabels()) {
			if (!weights.containsKey(label)) {
				weights.put(label, 0.0); 
				tps.put(label, 0.0);
				fps.put(label, 0.0);
			}
		}
		
		for (Pair<L, L> pair : actualAndPredicted) {
			L actual = pair.getFirst();
			weights.put(actual, weights.get(actual) + 1.0);
		}
		
		L filterLabelTyped = data.getDatumTools().labelFromString(this.filterLabel);
		for (Entry<L, Double> entry : weights.entrySet()) {
			if (filterLabelTyped != null) {
				if (entry.getKey().equals(filterLabelTyped))
					entry.setValue(1.0);
				else
					entry.setValue(0.0);
			} else if (this.weighted)
				entry.setValue(entry.getValue()/actualAndPredicted.size());
			else
				entry.setValue(1.0/weights.size());
		}
		
		for (Pair<L, L> pair : actualAndPredicted) {
			L actual = pair.getFirst();
			L predicted = pair.getSecond();
			
			if (actual.equals(predicted)) {
				tps.put(predicted, tps.get(predicted) + 1.0);
			} else {
				fps.put(predicted, fps.get(predicted) + 1.0);
			}
		}
		
		double precision = 0.0;
		for (Entry<L, Double> weightEntry : weights.entrySet()) {
			L label = weightEntry.getKey();
			double weight = weightEntry.getValue();
			double tp = tps.get(label);
			double fp = fps.get(label);
			
			if (tp == 0.0 && fp == 0.0)
				precision += weight;
			else
				precision += weight*tp/(tp + fp);
		}
		
		return precision;
	}

	@Override
	public String getGenericName() {
		return "Precision";
	}

	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public Obj getParameterValue(String parameter) {
		if (parameter.equals("weighted"))
			return Obj.stringValue(String.valueOf(this.weighted));
		else if (parameter.equals("filterLabel"))
			return Obj.stringValue((this.filterLabel == null) ? "" : this.filterLabel);
		else
			return null;
	}

	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		if (parameter.equals("weighted"))
			this.weighted = Boolean.valueOf(this.context.getMatchValue(parameterValue));
		else if (parameter.equals("filterLabel"))
			this.filterLabel = this.context.getMatchValue(parameterValue);
		else
			return false;
		return true;
	}

	@Override
	public SupervisedModelEvaluation<D, L> makeInstance(Context<D, L> context) {
		return new SupervisedModelEvaluationPrecision<D, L>(context);
	}
}
