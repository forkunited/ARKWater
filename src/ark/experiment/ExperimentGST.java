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

package ark.experiment;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;
import ark.data.feature.Feature;
import ark.model.SupervisedModel;
import ark.model.evaluation.GridSearchTestValidation;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.SerializationUtil;

/**
 * ExperimentGST represents a grid-search-test experiment which trains
 * a model on training data, searches for hyper-parameter values on dev
 * data, and finally (optionally) evaluates the model trained on
 * train+dev data with the best hyper-parameter values on test data.
 * An experiment configuration file determines the model, features, and
 * other settings for the experiment.  ExperimentGST parses this configuration
 * file with the help of the ark.model.SupervisedModel, ark.data.feature.Feature,
 * ark.util.SerializationUtil and other classes, and then uses 
 * ark.model.evaluation.GridSearchTestValidation to carry out the
 * experiment.  See the experiments/GSTTLinkType directory in the 
 * TemporalOrdering project at https://github.com/forkunited/TemporalOrdering
 * for examples of experiment configuration files.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 * 
 */
public class ExperimentGST<D extends Datum<L>, L> extends Experiment<D, L> {
	protected SupervisedModel<D, L> model;
	protected List<Feature<D, L>> features;
	protected String errorExampleExtractor;
	protected Map<String, List<String>> gridSearchParameterValues;
	protected List<SupervisedModelEvaluation<D, L>> evaluations;
	protected List<Double> evaluationValues;
	protected DataSet<D, L> trainData;
	protected DataSet<D, L> devData;
	protected DataSet<D, L> testData; // can be null
	protected boolean trainOnDev; // determines whether or not to retrain model on train+dev set
	protected Map<D, L> classifiedData;
	
	public ExperimentGST(String name, String inputPath, DataSet<D, L> trainData, DataSet<D, L> devData, DataSet<D, L> testData) {
		super(name, inputPath, trainData.getDatumTools());
		
		this.features = new ArrayList<Feature<D, L>>();
		this.gridSearchParameterValues = new HashMap<String, List<String>>();
		this.evaluations = new ArrayList<SupervisedModelEvaluation<D, L>>();
		this.trainData = trainData;
		this.devData = devData;
		this.testData = testData;
	}
	
	@Override
	protected boolean execute() {
		GridSearchTestValidation<D, L> gridSearchValidation = new GridSearchTestValidation<D, L>(
				this.name, 
				this.model, 
				this.features,
				this.maxThreads,
				this.trainData, 
				this.devData, 
				this.testData, 
				this.evaluations,
				this.trainOnDev);
		gridSearchValidation.setPossibleHyperParameterValues(this.gridSearchParameterValues);
		if (gridSearchValidation.run(this.errorExampleExtractor, true).get(0) < 0){
			this.classifiedData = gridSearchValidation.getClassifiedData();
			this.evaluationValues = gridSearchValidation.getEvaluationValues();
			return false;
		}
		this.classifiedData = gridSearchValidation.getClassifiedData();
		this.evaluationValues = gridSearchValidation.getEvaluationValues();
		return true;
	}
	
	@Override
	protected boolean deserializeNext(BufferedReader reader, String nextName) throws IOException {
		if (nextName.startsWith("model")) {
			String[] nameParts = nextName.split("_");
			String referenceName = null;
			if (nameParts.length > 1)
				referenceName = nameParts[1];
			
			String modelName = SerializationUtil.deserializeGenericName(reader);
			this.model = this.datumTools.makeModelInstance(modelName);
			if (!this.model.deserialize(reader, false, false, this.datumTools, referenceName))
				return false;
		} else if (nextName.startsWith("feature")) {
			String[] nameParts = nextName.split("_");
			String referenceName = null;
			boolean ignore = false;
			if (nameParts.length > 1)
				referenceName = nameParts[1];
			if (nameParts.length > 2)
				ignore = true;
			String featureName = SerializationUtil.deserializeGenericName(reader);
			Feature<D, L> feature = this.datumTools.makeFeatureInstance(featureName);
			if (!feature.deserialize(reader, false, false, this.datumTools, referenceName, ignore))
				return false;
			this.features.add(feature);
		} else if (nextName.startsWith("errorExampleExtractor")) {
			this.errorExampleExtractor = SerializationUtil.deserializeAssignmentRight(reader);
			
		} else if (nextName.startsWith("gridSearchParameterValues")) {
			String parameterName = SerializationUtil.deserializeGenericName(reader);
			List<String> parameterValues = SerializationUtil.deserializeList(reader);
			this.gridSearchParameterValues.put(parameterName, parameterValues);
		
		} else if (nextName.startsWith("evaluation")) {
			String evaluationName = SerializationUtil.deserializeGenericName(reader);
			SupervisedModelEvaluation<D, L> evaluation = this.datumTools.makeEvaluationInstance(evaluationName);
			if (!evaluation.deserialize(reader, false, this.datumTools))
				return false;
			this.evaluations.add(evaluation);
			
		} else if (nextName.startsWith("trainOnDev")) {
			this.trainOnDev = Boolean.valueOf(SerializationUtil.deserializeAssignmentRight(reader));
		}
		
		return true;
	}
	
	public Map<D, L> getClassifiedData(){
		if (this.classifiedData == null)
			throw new IllegalStateException("Trying to return classified data, but the data hasn't been classified yet.");
		else
			return this.classifiedData;
	}
	
	public List<Double> getEvaluationValues() {
		return this.evaluationValues;
	}
	
	public List<SupervisedModelEvaluation<D, L>> getEvaluations() {
		return this.evaluations;
	}
}
