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
import java.io.Writer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.model.evaluation.metric.SupervisedModelEvaluation;

/**
 * SupervisedModelLabelDistribution learns a single posterior
 * for all datums according to the label distribution in the
 * training data.  During classification, this leads the 
 * model to pick the majority baseline label.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public class SupervisedModelLabelDistribution<D extends Datum<L>, L> extends SupervisedModel<D, L> {
	private Map<L, Double> labelDistribution;
	
	public SupervisedModelLabelDistribution() {
		this.labelDistribution = new HashMap<L, Double>();
	}
	
	@Override
	public boolean train(FeaturizedDataSet<D, L> data, FeaturizedDataSet<D, L> testData, List<SupervisedModelEvaluation<D, L>> evaluations) {
		double total = 0;
		
		for (L label : this.validLabels)
			this.labelDistribution.put(label, 0.0);
		
		for (D datum : data) {
			L label = mapValidLabel(datum.getLabel());
			if (label == null)
				continue;
			
			this.labelDistribution.put(label, this.labelDistribution.get(label) + 1.0);
			total += 1.0;
		}
		
		for (Entry<L, Double> entry : this.labelDistribution.entrySet()) {
			entry.setValue(entry.getValue() / total);
		}

		return true;
	}

	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		Map<D, Map<L, Double>> posterior = new HashMap<D, Map<L, Double>>();
		for (D datum : data) {
			posterior.put(datum, this.labelDistribution);
		}
		return posterior;
	}
	
	@Override
	protected boolean deserializeParameters(BufferedReader reader,
			Tools<D, L> datumTools) {
		// TODO: Serialize the distribution.  This isn't necessary for now because we never save
		// this kind of model since it's just used to compute the majority baseline
		return true;
	}

	@Override
	protected boolean serializeParameters(Writer writer) {
		// TODO: Serialize the distribution.  This isn't necessary for now because we never save
		// this kind of model since it's just used to compute the majority baseline
		return true;
	}
	
	@Override
	protected String[] getHyperParameterNames() {
		return new String[0];
	}

	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelLabelDistribution<D, L>();
	}

	@Override
	protected boolean deserializeExtraInfo(String name, BufferedReader reader,
			Tools<D, L> datumTools) {
		return true;
	}

	@Override
	protected boolean serializeExtraInfo(Writer writer) {
		return true;
	}

	@Override
	public String getGenericName() {
		return "LabelDistribution";
	}

	@Override
	public String getHyperParameterValue(String parameter) {
		return null;
	}

	@Override
	public boolean setHyperParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		return true;
	}
}
