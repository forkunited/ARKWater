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

import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
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

	@Override
	protected double compute(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data, Map<D, L> predictions) {
		List<Pair<L, L>> actualAndPredicted = getMappedActualAndPredictedLabels(predictions);
		double total = actualAndPredicted.size();
		double correct = 0;
		for (Pair<L, L> pair : actualAndPredicted) {
			if (pair.getFirst().equals(pair.getSecond()))
				correct += 1.0;
		}
		
		return (total == 0) ? 1 : correct/total;
	}

	@Override
	public String getGenericName() {
		return "Accuracy";
	}

	@Override
	protected String[] getParameterNames() {
		return new String[0];
	}

	@Override
	protected String getParameterValue(String parameter) {
		return null;
	}

	@Override
	protected boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		return true;
	}

	@Override
	protected SupervisedModelEvaluation<D, L> makeInstance() {
		return new SupervisedModelEvaluationAccuracy<D, L>();
	}

}
