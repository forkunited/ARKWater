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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelMapping;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.util.Pair;
import ark.util.Parameterizable;
import ark.util.SerializationUtil;

/**
 * SupervisedModelEvaluation represents evaluation measure
 * for a supervised classification model.  
 * 
 * Implementations of particular evaluation measures derive
 * from SupervisedModelEvaluation, and SupervisedModelEvaluation
 * is primarily responsible for providing the methods necessary
 * for deserializing these evaluation measures from configuration
 * files.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public abstract class SupervisedModelEvaluation<D extends Datum<L>, L> implements Parameterizable<D, L> {
	protected LabelMapping<L> labelMapping;
	
	/**
	 * @return a generic name by which to refer to the evaluation
	 */
	public abstract String getGenericName();
	
	/**
	 * 
	 * @param model
	 * @param data
	 * @param predictions
	 * @return the value of the evaluation measure for the model with predictions on the
	 * given data set.  This method should generally start by calling 
	 * 'getMappedActualAndPredictedLabels' to get the labels according to the 'labelMapping'
	 * function, and then compute the measure using those returned labels.
	 * 
	 */
	protected abstract double compute(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data, Map<D, L> predictions);
	
	/**
	 * @return a generic instance of the evaluation measure.  This is used when deserializing
	 * the parameters for the measure from a configuration file
	 */
	public abstract SupervisedModelEvaluation<D, L> makeInstance();
	
	/**
	 * @param predictions
	 * @return a list of pairs of actual predicted labels that are mapped by
	 * the label mapping (labelMapping) from the given predictions map
	 */
	protected List<Pair<L, L>> getMappedActualAndPredictedLabels(Map<D, L> predictions) {
		List<Pair<L, L>> mappedLabels = new ArrayList<Pair<L, L>>(predictions.size());
		for (Entry<D, L> prediction : predictions.entrySet()) {
			L actual = prediction.getKey().getLabel();
			L predicted = prediction.getValue();
			if (this.labelMapping != null) {
				actual = this.labelMapping.map(actual);
				predicted = this.labelMapping.map(predicted);
			}
			
			Pair<L, L> mappedPair = new Pair<L, L>(actual, predicted);
			mappedLabels.add(mappedPair);
		}
		
		return mappedLabels;
	}
	
	/**
	 * 
	 * @param model
	 * @param data
	 * @param predictions
	 * @return the value of the evaluation measure for the model with predictions on the
	 * given data set.  If the model has a label mapping, then it is used when 
	 * computing the evaluation measure.
	 */
	public double evaluate(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data, Map<D, L> predictions) {
		LabelMapping<L> modelLabelMapping = model.getLabelMapping();
		if (this.labelMapping != null)
			model.setLabelMapping(this.labelMapping);
		
		double evaluation = compute(model, data, predictions);
		model.setLabelMapping(modelLabelMapping);
		
		return evaluation;
	}
	
	public <D1 extends Datum<L1>, L1> SupervisedModelEvaluation<D1, L1> clone(Datum.Tools<D1, L1> datumTools) {
		return clone(datumTools, null, true);
	}
	
	@SuppressWarnings("unchecked")
	public <D1 extends Datum<L1>, L1> SupervisedModelEvaluation<D1, L1> clone(Datum.Tools<D1, L1> datumTools, Map<String, String> environment, boolean copyLabelMapping) {
		SupervisedModelEvaluation<D1, L1> clone = datumTools.makeEvaluationInstance(getGenericName(), true);
		String[] parameterNames = getParameterNames();
		for (int i = 0; i < parameterNames.length; i++) {
			String parameterValue = getParameterValue(parameterNames[i]);
			if (environment != null && parameterValue != null) {
				for (Entry<String, String> entry : environment.entrySet())
					parameterValue = parameterValue.replace("--" + entry.getKey() + "--", entry.getValue());
			}
			clone.setParameterValue(parameterNames[i], parameterValue, datumTools);
		}
		
		if (copyLabelMapping)
			clone.labelMapping = (LabelMapping<L1>)this.labelMapping;
		
		return clone;
	}
	
	public boolean deserialize(BufferedReader reader, boolean readGenericName, Datum.Tools<D, L> datumTools) throws IOException {
		if (readGenericName && SerializationUtil.deserializeGenericName(reader) == null)
			return false;
		
		Map<String, String> parameters = SerializationUtil.deserializeArguments(reader);
		if (parameters != null)
			for (Entry<String, String> entry : parameters.entrySet()) {
				if (entry.getKey().equals("labelMapping"))
					this.labelMapping = datumTools.getLabelMapping(entry.getValue());
				else
					setParameterValue(entry.getKey(), entry.getValue(), datumTools);
			}

		return true;
	}
	
	public boolean serialize(Writer writer) throws IOException {
		writer.write(toString(false));
		return true;
	}
	
	public String toString(boolean withVocabulary) {
		if (withVocabulary) {
			StringWriter stringWriter = new StringWriter();
			try {
				if (serialize(stringWriter))
					return stringWriter.toString();
				else
					return null;
			} catch (IOException e) {
				return null;
			}
		} else {
			String genericName = getGenericName();
			Map<String, String> parameters = new HashMap<String, String>();
			String[] parameterNames = getParameterNames();
			for (int i = 0; i < parameterNames.length; i++)
				parameters.put(parameterNames[i], getParameterValue(parameterNames[i]));
			
			if (this.labelMapping != null)
				parameters.put("labelMapping", this.labelMapping.toString());
			
			StringWriter parametersWriter = new StringWriter();
			
			try {
				SerializationUtil.serializeArguments(parameters, parametersWriter);
			} catch (IOException e) {
				return null;
			}
			
			String parametersStr = parametersWriter.toString();
			return genericName + "(" + parametersStr + ")";
		}
	}
	
	public String toString() {
		return toString(false);
	}
	
	
	public boolean fromString(String str, Datum.Tools<D, L> datumTools) {
		try {
			return deserialize(new BufferedReader(new StringReader(str)), true, datumTools);
		} catch (IOException e) {
			
		}
		return true;
	}
	
}
