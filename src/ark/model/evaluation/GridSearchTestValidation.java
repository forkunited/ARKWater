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

import ark.data.annotation.Datum;
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
	private String name;
	private SupervisedModel<D, L> model;
	private FeaturizedDataSet<D, L> trainData;
	private FeaturizedDataSet<D, L> devData;
	private FeaturizedDataSet<D, L> testData;
	private ConfusionMatrix<D, L> confusionMatrix;
	private Map<String, List<String>> possibleParameterValues; 
	private List<SupervisedModelEvaluation<D, L>> evaluations;
	private List<Double> evaluationValues;
	private DecimalFormat cleanDouble;
	private List<GridSearch<D, L>.EvaluatedGridPosition> gridEvaluation;
	private GridSearch<D,L>.EvaluatedGridPosition bestGridPosition;
	private boolean trainOnDev;
	private Map<D, L> classifiedData;
	
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
							  FeaturizedDataSet<D, L> trainData,
							  FeaturizedDataSet<D, L> devData,
							  FeaturizedDataSet<D, L> testData,
							  List<SupervisedModelEvaluation<D, L>> evaluations,
							  boolean trainOnDev) {
		this.name = name;
		this.model = model;
		this.trainData = trainData;
		this.devData = devData;
		this.testData = testData;
		this.evaluations = evaluations;
		this.possibleParameterValues = new HashMap<String, List<String>>();
		this.gridEvaluation = new ArrayList<GridSearch<D, L>.EvaluatedGridPosition>();
		this.cleanDouble = new DecimalFormat("0.00");
		this.trainOnDev = trainOnDev;
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
	
	public List<Double> run(Datum.Tools.TokenSpanExtractor<D, L> errorExampleExtractor, boolean outputResults) {
		return run(errorExampleExtractor, outputResults, 1);
	}
	
	public List<Double> run(Datum.Tools.TokenSpanExtractor<D, L> errorExampleExtractor, boolean outputResults, int maxThreads) {
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
			this.bestGridPosition = gridSearch.getBestPosition(maxThreads);
			this.gridEvaluation = gridSearch.getGridEvaluation(maxThreads);
				
			output.debugWriteln("Grid search (" + this.name + ": \n" + gridSearch.toString());
				
			if (this.bestGridPosition != null) {
				this.model.setHyperParameterValues(this.bestGridPosition.getCoordinates(), this.trainData.getDatumTools());
				this.model = this.model.clone(this.trainData.getDatumTools());
			}
		}
		long totalGridSearchTime = System.currentTimeMillis() - startGridSearch;
			
		output.debugWriteln("Training model with best parameters (" + this.name + ")");
		
		Pair<Long, Long> trainAndTestTime;
		
		evaluateConstraints(this.trainData, this.devData, this.testData);
		
		this.evaluationValues = null;
		if (this.testData != null) {
			if (this.trainOnDev)
				this.trainData.addAll(this.devData);
	
			TrainTestValidation<D, L> accuracy = new TrainTestValidation<D, L>(this.name, this.model, this.trainData, this.testData, this.evaluations);
			this.evaluationValues = accuracy.run();
			if (this.evaluationValues.get(0) < 0) {
				output.debugWriteln("Error: Validation failed (" + this.name + ")");
				return null;
			} 
			
			this.classifiedData = accuracy.getClassifiedData();
			this.confusionMatrix = accuracy.getConfusionMatrix();
			output.debugWriteln("Test " + this.evaluations.get(0).toString() + " (" + this.name + ": " + cleanDouble.format(this.evaluationValues.get(0)));
			
			trainAndTestTime = accuracy.getTrainAndTestTime();
		} else {
			this.evaluationValues = this.bestGridPosition.getValidation().getResults();
			this.classifiedData = this.bestGridPosition.getValidation().getClassifiedData();
			this.confusionMatrix = this.bestGridPosition.getValidation().getConfusionMatrix();
			output.debugWriteln("Dev best " + this.evaluations.get(0).toString() + " (" + this.name + ": " + cleanDouble.format(this.evaluationValues.get(0)));
			this.model = this.bestGridPosition.getValidation().getModel();
			trainAndTestTime = this.bestGridPosition.getValidation().getTrainAndTestTime();
		}
		
		output.dataWriteln(this.confusionMatrix.getActualToPredictedDescription(errorExampleExtractor));
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
	
	private void evaluateConstraints(FeaturizedDataSet<D, L> trainData2,
			FeaturizedDataSet<D, L> devData2, FeaturizedDataSet<D, L> testData2) {
		
		
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
		return String.format("%d:%d:%d", 
				TimeUnit.MILLISECONDS.toHours(duration),
			    TimeUnit.MILLISECONDS.toMinutes(duration) - TimeUnit.HOURS.toMinutes(TimeUnit.MILLISECONDS.toHours(duration)),
			    TimeUnit.MILLISECONDS.toSeconds(duration) - TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(duration) - TimeUnit.HOURS.toMinutes(TimeUnit.MILLISECONDS.toHours(duration))) - 
			    	TimeUnit.HOURS.toSeconds(TimeUnit.MILLISECONDS.toHours(duration)));
	}
	
	
}
