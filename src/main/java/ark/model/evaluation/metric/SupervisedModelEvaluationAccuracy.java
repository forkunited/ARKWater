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

import ark.data.Context;
import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.parse.Obj;
import ark.util.Pair;

/**
 * SupervisedModelEvaluationAccuracy computes the micro-averaged accuracy
 * for a supervised classification model on a data set.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public class SupervisedModelEvaluationAccuracy<D extends Datum<L>, L> extends SupervisedModelEvaluation<D, L> {
	private boolean computeBaseline;
	private String[] parameterNames = { "computeBaseline" };
	
	public SupervisedModelEvaluationAccuracy() {
		
	}
	
	public SupervisedModelEvaluationAccuracy(Context<D, L> context) {
		this.context = context;
	}
	
	@Override
	protected double compute(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data, Map<D, L> predictions) {
		List<Pair<L, L>> actualAndPredicted = getMappedActualAndPredictedLabels(predictions);
		Map<L, Integer> actualLabelCounts = new HashMap<L, Integer>();
		double total = actualAndPredicted.size();
		double correct = 0;
		for (Pair<L, L> pair : actualAndPredicted) {
			if (this.computeBaseline) {
				if (!actualLabelCounts.containsKey(pair.getFirst()))
					actualLabelCounts.put(pair.getFirst(), 0);
				actualLabelCounts.put(pair.getFirst(), actualLabelCounts.get(pair.getFirst()) + 1);
			} else if (pair.getFirst().equals(pair.getSecond()))
				correct += 1.0;
		}
		
		if (this.computeBaseline) {
			int maxCount = 0;
			for (Integer labelCount : actualLabelCounts.values())
				if (labelCount > maxCount)
					maxCount = labelCount;
			return (total == 0) ? 0 : maxCount/total; 
		} else {
			return (total == 0) ? 0 : correct/total;
		}
	}

	@Override
	public String getGenericName() {
		return "Accuracy";
	}

	@Override
	public SupervisedModelEvaluation<D, L> makeInstance(Context<D, L> context) {
		return new SupervisedModelEvaluationAccuracy<D, L>(context);
	}

	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public Obj getParameterValue(String parameter) {
		if (parameter.equals("computeBaseline"))
			return Obj.stringValue(String.valueOf(this.computeBaseline));
		else
			return null;
	}

	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		if (parameter.equals("computeBaseline"))
			this.computeBaseline = Boolean.valueOf(this.context.getMatchValue(parameterValue));
		else
			return false;
		return true;
	}
}
