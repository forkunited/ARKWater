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

package ark.model.evaluation;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.OutputWriter;
import ark.util.Pair;

/**
 * TrainTestValidation trains a model on a training data
 * set and evaluates it on a test data set by given 
 * evaluation metrics
 * 
 * @author Bill McDowell
 *
 */
public class TrainTestValidation<D extends Datum<L>, L> {
	private String name;
	private SupervisedModel<D, L> model;
	private FeaturizedDataSet<D, L> trainData;
	private FeaturizedDataSet<D, L> testData;
	private ConfusionMatrix<D, L> confusionMatrix;
	private List<SupervisedModelEvaluation<D, L>> evaluations;
	private List<Double> results; // Results of evaluations
	private Pair<Long, Long> trainAndTestTime;
	private Map<D, L> classifiedData;
	
	/**
	 * 
	 * @param name
	 * @param model
	 * @param trainData 
	 * @param testData
	 * @param evaluations - Measures by which to evaluate the model
	 */
	public TrainTestValidation(String name,
							  SupervisedModel<D, L> model, 
							  FeaturizedDataSet<D, L> trainData,
							  FeaturizedDataSet<D, L> testData,
							  List<SupervisedModelEvaluation<D, L>> evaluations) {
		this.name = name;
		this.model = model;
		this.trainData = trainData;
		this.testData = testData;
		this.evaluations = evaluations;
		this.results = new ArrayList<Double>(this.evaluations.size());
	}
	
	public List<Double> run() {
		return run(false);
	}
	
	/**
	 * @param skipTraining indicates whether to skip model training
	 * @return trains and evaluates the model, returning a list of values as results
	 * of the evaluations
	 */
	public List<Double> run(boolean skipTraining) {
		OutputWriter output = this.trainData.getDatumTools().getDataTools().getOutputWriter();
		
		for (int i = 0; i < this.evaluations.size(); i++)
			this.results.add(-1.0);
		
		Long totalTrainTime = (long)0;
		if (!skipTraining) {
			output.debugWriteln("Training model (" + this.name + ")");
			Long startTrain = System.currentTimeMillis();
			if (!this.model.train(this.trainData, this.testData, this.evaluations))
				return this.results;
			totalTrainTime = System.currentTimeMillis() - startTrain;
		}
		output.debugWriteln("Classifying data (" + this.name + ")");
		
		Long startTest = System.currentTimeMillis();
		Map<D, L> classifiedData = this.model.classify(this.testData);
		if (classifiedData == null)
			return this.results;
		Long totalTestTime = System.currentTimeMillis() - startTest;
		this.trainAndTestTime = new Pair<Long, Long>(totalTrainTime, totalTestTime);
		this.classifiedData = classifiedData;
		
		output.debugWriteln("Computing model score (" + this.name + ")");
		
		for (int i = 0; i < this.evaluations.size(); i++)
			this.results.set(i, this.evaluations.get(i).evaluate(this.model, this.testData, classifiedData));
		
		this.confusionMatrix = new ConfusionMatrix<D, L>(this.model.getValidLabels(), this.model.getLabelMapping());
		this.confusionMatrix.addData(classifiedData);
		
		return this.results;
	}
	
	public List<SupervisedModelEvaluation<D, L>> getEvaluations() {
		return this.evaluations;
	}
	
	public ConfusionMatrix<D, L> getConfusionMatrix() {
		return this.confusionMatrix;
	}
	
	public SupervisedModel<D, L> getModel() {
		return this.model;
	}
	
	public List<Double> getResults() {
		return this.results;
	}
	
	public Pair<Long, Long> getTrainAndTestTime(){
		return this.trainAndTestTime;
	}
	
	public Map<D, L> getClassifiedData(){
		if (this.classifiedData == null)
			throw new IllegalStateException("Trying to return classified data, but the data hasn't been classified yet.");
		else
			return this.classifiedData;
	}
}