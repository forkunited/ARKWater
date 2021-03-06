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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.Context;
import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelIndicator;
import ark.data.annotation.Datum.Tools.LabelMapping;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.parse.ARKParsableFunction;
import ark.parse.Assignment;
import ark.parse.AssignmentList;
import ark.parse.Obj;
import ark.util.Pair;

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
public abstract class SupervisedModelEvaluation<D extends Datum<L>, L> extends ARKParsableFunction {
	protected Context<D, L> context;
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
	public abstract SupervisedModelEvaluation<D, L> makeInstance(Context<D, L> context);
	
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
		LabelMapping<L> modelLabelMapping = null;
		if (model != null)
			model.getLabelMapping();
		if (this.labelMapping != null)
			model.setLabelMapping(this.labelMapping);
		
		double evaluation = compute(model, data, predictions);
		
		if (model != null)
			model.setLabelMapping(modelLabelMapping);
		
		return evaluation;
	}
		
	public SupervisedModelEvaluation<D, L> clone() {
		SupervisedModelEvaluation<D, L> clone = this.context.getDatumTools().makeEvaluationInstance(getGenericName(), this.context);
		if (!clone.fromParse(getModifiers(), getReferenceName(), toParse()))
			return null;
		return clone;
	}
	
	public <T extends Datum<Boolean>> SupervisedModelEvaluation<T, Boolean> makeBinary(Context<T, Boolean> context, LabelIndicator<L> labelIndicator) {
		SupervisedModelEvaluation<T, Boolean> binaryEvaluation = context.getDatumTools().makeEvaluationInstance(getGenericName(), context);
		
		if (binaryEvaluation == null)
			return null;
		
		binaryEvaluation.referenceName = this.referenceName;
		binaryEvaluation.modifiers = this.modifiers;
		binaryEvaluation.labelMapping = null;
		
		String[] parameterNames = getParameterNames();
		for (int i = 0; i < parameterNames.length; i++)
			binaryEvaluation.setParameterValue(parameterNames[i], getParameterValue(parameterNames[i]));
		
		return binaryEvaluation;
	}
	
	@Override
	protected boolean fromParseInternal(AssignmentList internalAssignments) {
		if (internalAssignments == null)
			return true;
		
		if (internalAssignments.contains("labelMapping")) {
			Obj.Value labelMapping = (Obj.Value)internalAssignments.get("labelMapping").getValue();
			this.labelMapping = this.context.getDatumTools().getLabelMapping(this.context.getMatchValue(labelMapping));
		}
		
		return true;
	}
	
	@Override
	protected AssignmentList toParseInternal() {
		AssignmentList internalAssignments = new AssignmentList();
		
		if (this.labelMapping != null) {
			Obj.Value labelMapping = Obj.stringValue(this.labelMapping.toString());
			internalAssignments.add(Assignment.assignmentTyped(new ArrayList<String>(), Context.VALUE_STR, "labelMapping", labelMapping));
		}
		
		return (internalAssignments.size() == 0) ? null : internalAssignments;
	}
}
