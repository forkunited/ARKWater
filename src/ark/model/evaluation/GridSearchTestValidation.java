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

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.TimeUnit;

import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;
import ark.data.feature.Feature;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.OutputWriter;
import ark.util.Pair;

/**
 * GridSearchTestValidation performs a grid search for model
 * hyper-parameter values using a training and dev data sets.
 * Then, it sets the hyper-parameters to the best values from
 * the grid search, retrains on the train+dev data, and
 * gives a final evaluation on the test data. 
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public class GridSearchTestValidation<D extends Datum<L>, L> {
	protected String name;
	protected SupervisedModel<D, L> model;
	protected int maxThreads;
	protected FeaturizedDataSet<D, L> trainData;
	protected FeaturizedDataSet<D, L> devData;
	protected FeaturizedDataSet<D, L> testData;
	protected ConfusionMatrix<D, L> confusionMatrix;
	protected Map<String, List<String>> possibleParameterValues; 
	protected List<SupervisedModelEvaluation<D, L>> evaluations;
	protected List<Double> evaluationValues;
	protected DecimalFormat cleanDouble;
	protected List<GridSearch<D, L>.EvaluatedGridPosition> gridEvaluation;
	protected GridSearch<D,L>.EvaluatedGridPosition bestGridPosition;
	protected boolean trainOnDev;
	protected Map<D, L> classifiedData;
	
	/**
	 * 
	 * @param name - Name for the validation used in debug output strings
	 * @param model
	 * @param trainData
	 * @param devData
	 * @param testData - Data on which to give the final evaluation.  This 
	 * can be null if you want to only do the grid search without any final 
	 * evaluation
	 * 
	 * @param evaluations - Measures by which to evaluate the model.  The
	 * first of these is used for the grid search
	 * 
	 * @param trainOnDev - Indicates whether or not to use the dev data 
	 * for training in the final evaluation
	 */
	public GridSearchTestValidation(String name,
							  SupervisedModel<D, L> model, 
							  int maxThreads,
							  FeaturizedDataSet<D, L> trainData,
							  FeaturizedDataSet<D, L> devData,
							  FeaturizedDataSet<D, L> testData,
							  List<SupervisedModelEvaluation<D, L>> evaluations,
							  boolean trainOnDev) {
		this.name = name;
		this.model = model;
		this.maxThreads = maxThreads;
		this.trainData = trainData;
		this.devData = devData;
		this.testData = testData;
		this.evaluations = evaluations;
		this.possibleParameterValues = new HashMap<String, List<String>>();
		this.gridEvaluation = new ArrayList<GridSearch<D, L>.EvaluatedGridPosition>();
		this.cleanDouble = new DecimalFormat("0.00000");
		this.trainOnDev = trainOnDev;
	}
	
	public GridSearchTestValidation(String name,
			  SupervisedModel<D, L> model, 
			  List<Feature<D, L>> features,
			  int maxThreads,
			  DataSet<D, L> trainData,
			  DataSet<D, L> devData,
			  DataSet<D, L> testData,
			  List<SupervisedModelEvaluation<D, L>> evaluations,
			  boolean trainOnDev) {
		this(name, model, maxThreads,
				new FeaturizedDataSet<D, L>(name + " Training", maxThreads, trainData.getDatumTools(), trainData.getLabelMapping()), 
				new FeaturizedDataSet<D, L>(name + " Dev", maxThreads, devData.getDatumTools(), devData.getLabelMapping()), 
				(testData == null) ? null : new FeaturizedDataSet<D, L>(name + " Test", maxThreads, testData.getDatumTools(), testData.getLabelMapping()), 
				evaluations, trainOnDev);
		
		OutputWriter output = this.trainData.getDatumTools().getDataTools().getOutputWriter();
		this.trainData.addAll(trainData);
		this.devData.addAll(devData);
		if (this.testData != null)
			this.testData.addAll(testData);
		
		output.debugWriteln("Initializing features (" + this.name + ")...");
		if (!this.trainData.addFeatures(features, true)) {
			output.debugWriteln("ERROR: Failed to initialize features.");
			return;
		}
		
		this.devData.addFeatures(features, false);
		if (testData != null)
			this.testData.addFeatures(features, false);
		
		output.debugWriteln("Finished initializing features (" + this.name + ").");
		
		output.debugWriteln("Outputting initalized features (" + this.name + ")...");
		
		for (Feature<D, L> feature : features)
			output.modelWriteln(feature.toString(true));
		output.debugWriteln("Finished outputting initalized features (" + this.name + ").");
	}
	
	public boolean addPossibleHyperParameterValue(String parameter, String parameterValue) {
		if (!this.possibleParameterValues.containsKey(parameter))
			this.possibleParameterValues.put(parameter, new ArrayList<String>());
		this.possibleParameterValues.get(parameter).add(parameterValue);
		
		return true;
	}
	
	public boolean setPossibleHyperParameterValues(Map<String, List<String>> possibleParameterValues) {
		this.possibleParameterValues = possibleParameterValues;
		return true;
	}
	
	public List<Double> run(String errorExampleExtractor, boolean outputResults) {
		long startTime = System.currentTimeMillis();
		OutputWriter output = this.trainData.getDatumTools().getDataTools().getOutputWriter();
		
		long startGridSearch = System.currentTimeMillis();
		if (this.possibleParameterValues.size() > 0) {
			GridSearch<D, L> gridSearch = new GridSearch<D,L>(this.name,
										this.model,
						 				this.trainData, 
						 				this.devData,
						 				this.possibleParameterValues,
						 				this.evaluations.get(0)); 
			this.bestGridPosition = gridSearch.getBestPosition(this.maxThreads);
			this.gridEvaluation = gridSearch.getGridEvaluation(this.maxThreads);
				
			output.debugWriteln("Grid search (" + this.name + "): \n" + gridSearch.toString());
				
			if (this.bestGridPosition != null) {
				if (!this.trainOnDev) 
					this.model = this.bestGridPosition.getValidation().getModel();
				else {
					this.model.setHyperParameterValues(this.bestGridPosition.getCoordinates(), this.trainData.getDatumTools());
					this.model = this.model.clone(this.trainData.getDatumTools());
				}
			}
		}
		long totalGridSearchTime = System.currentTimeMillis() - startGridSearch;
			
		output.debugWriteln("Train and/or evaluating model with best parameters (" + this.name + ")");
		
		Pair<Long, Long> trainAndTestTime;
		
		this.evaluationValues = null;
		if (this.testData != null) {
			if (this.trainOnDev) 
				this.trainData.addAll(this.devData); // FIXME Reinitialize features on train+dev?
	
			TrainTestValidation<D, L> accuracy = new TrainTestValidation<D, L>(this.name, this.model, this.trainData, this.testData, this.evaluations);
			this.evaluationValues = accuracy.run(!this.trainOnDev);
			if (this.evaluationValues.get(0) < 0) {
				output.debugWriteln("Error: Validation failed (" + this.name + ")");
				return null;
			} 
			
			this.classifiedData = accuracy.getClassifiedData();
			this.confusionMatrix = accuracy.getConfusionMatrix();
			output.debugWriteln("Test " + this.evaluations.get(0).toString() + " (" + this.name + "): " + cleanDouble.format(this.evaluationValues.get(0)));
			
			trainAndTestTime = accuracy.getTrainAndTestTime();
		} else {
			this.evaluationValues = this.bestGridPosition.getValidation().getResults();
			this.classifiedData = this.bestGridPosition.getValidation().getClassifiedData();
			this.confusionMatrix = this.bestGridPosition.getValidation().getConfusionMatrix();
			output.debugWriteln("Dev best " + this.evaluations.get(0).toString() + " (" + this.name + "): " + cleanDouble.format(this.evaluationValues.get(0)));
			this.model = this.bestGridPosition.getValidation().getModel();
			trainAndTestTime = this.bestGridPosition.getValidation().getTrainAndTestTime();
		}
		
		output.dataWriteln(this.confusionMatrix.getActualToPredictedDescription(this.trainData.getDatumTools().getTokenSpanExtractor(errorExampleExtractor)));
		output.modelWriteln(this.model.toString());
				
		if (outputResults) {
			if (this.bestGridPosition != null) {
				Map<String, String> parameters = this.bestGridPosition.getCoordinates();
				output.resultsWriteln("Best parameters from grid search:");
				for (Entry<String, String> entry : parameters.entrySet())
					output.resultsWriteln(entry.getKey() + ": " + entry.getValue());
			}
			
			if (this.testData != null) {
				output.resultsWriteln("\nTest set evaluation results: ");
			} else {
				output.resultsWriteln("\nDev set best evaluation results: ");
			}
			
			for (int i = 0; i < this.evaluations.size(); i++)
				output.resultsWriteln(this.evaluations.get(i).toString() + ": " + evaluationValues.get(i));
			
			output.resultsWriteln("\nConfusion matrix:\n" + this.confusionMatrix.toString());
			
			if (this.gridEvaluation != null && this.gridEvaluation.size() > 0) {
				output.resultsWriteln("\nGrid search on " + this.evaluations.get(0).toString() + ":");
				output.resultsWriteln(this.gridEvaluation.get(0).toKeyString("\t") + "\t" + this.evaluations.get(0).toString());
				for (GridSearch<D, L>.EvaluatedGridPosition gridPosition : this.gridEvaluation) {
					output.resultsWriteln(gridPosition.toValueString("\t") + "\t" + gridPosition.getPositionValue());
				}
			}
		}

		long totalTime = System.currentTimeMillis() - startTime;
		writeTimers(output, trainAndTestTime, totalGridSearchTime, totalTime);
		
		return evaluationValues;
	}

	public List<SupervisedModelEvaluation<D, L>> getEvaluations() {
		return this.evaluations;
	}
	
	public ConfusionMatrix<D, L> getConfusionMatrix() {
		return this.confusionMatrix;
	}
	
	public List<GridSearch<D, L>.EvaluatedGridPosition> getGridEvaluation() {
		return this.gridEvaluation;
	}
	
	public GridSearch<D, L>.EvaluatedGridPosition getBestGridPosition() {
		return this.bestGridPosition;
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
	
	// Outputting the timing info to the debug file.
	public void writeTimers(OutputWriter output, Pair<Long, Long> trainAndTestTime, long totalGridSearchTime, long totalTime){
		output.debugWriteln("");
		output.debugWriteln("The times for running the experiment, in \"HOURS:MINUTES:SECONDS\" format:");
		output.debugWriteln("The total train time: " + formatTime(trainAndTestTime.getFirst()));
		output.debugWriteln("The total test time: " + formatTime(trainAndTestTime.getSecond()));
		output.debugWriteln("The total grid search time: " + formatTime(totalGridSearchTime));
		output.debugWriteln("The total from start to finish: " + formatTime(totalTime));
	}
	
	public String formatTime(long duration){
		return String.format("%d h %d m %d s", 
				TimeUnit.MILLISECONDS.toHours(duration),
			    TimeUnit.MILLISECONDS.toMinutes(duration) - TimeUnit.HOURS.toMinutes(TimeUnit.MILLISECONDS.toHours(duration)),
			    TimeUnit.MILLISECONDS.toSeconds(duration) - TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(duration) - TimeUnit.HOURS.toMinutes(TimeUnit.MILLISECONDS.toHours(duration))) - 
			    	TimeUnit.HOURS.toSeconds(TimeUnit.MILLISECONDS.toHours(duration)));
	}
	
	
}
