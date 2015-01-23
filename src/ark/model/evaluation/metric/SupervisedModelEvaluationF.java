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

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.util.Pair;

/**
 * SupervisedModelEvaluationF computes an F measure
 * (http://en.wikipedia.org/wiki/F1_score)
 * for a supervised classification model on a data set.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public class SupervisedModelEvaluationF<D extends Datum<L>, L> extends SupervisedModelEvaluation<D, L> {
	/**
	 * Mode determines whether the F measure should be macro-averaged, micro-averaged,
	 * or macro-averaged weighted by actual label frequencies.
	 *
	 */
	public enum Mode {
		MACRO,
		MACRO_WEIGHTED,
		MICRO
	}
	
	private Mode mode = Mode.MACRO_WEIGHTED;
	private double Beta = 1.0;
	private L filterLabel;
	private String[] parameterNames = { "mode", "Beta", "filterLabel" };
	
	@Override
	protected double compute(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data, Map<D, L> predictions) {
		if (this.mode == Mode.MICRO)
			return computeMicro(model, data, predictions);
		else
			return computeMacro(model, data, predictions);
	}

	// Equal to micro accuracy...
	private double computeMicro(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data, Map<D, L> predictions) {
		List<Pair<L, L>> actualAndPredicted = this.getMappedActualAndPredictedLabels(predictions);
		double tp = 0;
		double f = 0; // fp or fn
		
		for (Pair<L, L> pair : actualAndPredicted) {
			if (pair.getFirst().equals(pair.getSecond()))
				tp++;
			else {
				f++;
			}
		}
		
		double pr = tp/(tp + f);
		return pr;//2*pr*pr/(pr+pr);
	}
	
	private double computeMacro(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data, Map<D, L> predictions) {
		List<Pair<L, L>> actualAndPredicted = this.getMappedActualAndPredictedLabels(predictions);
		Map<L, Double> weights = new HashMap<L, Double>();
		Map<L, Double> tps = new HashMap<L, Double>();
		Map<L, Double> fps = new HashMap<L, Double>();
		Map<L, Double> fns = new HashMap<L, Double>();
		
		for (L label : model.getValidLabels()) {
			if (!weights.containsKey(label)) {
				weights.put(label, 0.0); 
				tps.put(label, 0.0);
				fps.put(label, 0.0);
				fns.put(label, 0.0);
			}
		}
		
		for (Pair<L, L> pair : actualAndPredicted) {
			L actual = pair.getFirst();
			weights.put(actual, weights.get(actual) + 1.0);
		}
		
		for (Entry<L, Double> entry : weights.entrySet()) {
			if (this.filterLabel != null) {
				if (entry.getKey().equals(this.filterLabel))
					entry.setValue(1.0);
				else
					entry.setValue(0.0);
			} else if (this.mode == Mode.MACRO_WEIGHTED)
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
				fns.put(actual, fns.get(actual) + 1.0);
			}
		}
		
		double F = 0.0;
		double Beta2 = this.Beta*this.Beta;
		for (Entry<L, Double> weightEntry : weights.entrySet()) {
			L label = weightEntry.getKey();
			double weight = weightEntry.getValue();
			double tp = tps.get(label);
			double fp = fps.get(label);
			double fn = fns.get(label);
			
			if (tp == 0.0 && fn == 0.0 && fp == 0.0)
				F += weight;
			else
				F += weight*(1.0+Beta2)*tp/((1.0+Beta2)*tp + Beta2*fn + fp);
		}
		
		return F;
	}
	
	@Override
	public String getGenericName() {
		return "F";
	}

	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public String getParameterValue(String parameter) {
		if (parameter.equals("mode"))
			return this.mode.toString();
		else if (parameter.equals("Beta"))
			return String.valueOf(this.Beta);
		else if (parameter.equals("filterLabel"))
			return (this.filterLabel == null) ? "" : this.filterLabel.toString();
		else
			return null;
	}

	@Override
	public boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("mode"))
			this.mode = Mode.valueOf(parameterValue);
		else if (parameter.equals("Beta"))
			this.Beta = Double.valueOf(parameterValue);
		else if (parameter.equals("filterLabel"))
			this.filterLabel = datumTools.labelFromString(parameterValue);
		else
			return false;
		return true;
	}

	@Override
	public SupervisedModelEvaluation<D, L> makeInstance() {
		return new SupervisedModelEvaluationF<D, L>();
	}
}
