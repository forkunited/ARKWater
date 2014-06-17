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

public class GridSearchTestValidation<D extends Datum<L>, L> {
	private String name;
	private SupervisedModel<D, L> model;
	private FeaturizedDataSet<D, L> trainData;
	private FeaturizedDataSet<D, L> devData;
	private FeaturizedDataSet<D, L> testData;
	private ConfusionMatrix<D, L> confusionMatrix;
	private Map<String, List<String>> possibleParameterValues; 
	private List<SupervisedModelEvaluation<D, L>> evaluations;
	private DecimalFormat cleanDouble;
	private List<GridSearch<D, L>.EvaluatedGridPosition> gridEvaluation;
	private GridSearch<D,L>.EvaluatedGridPosition bestGridPosition;
	private boolean trainOnDev;
	private Map<D, L> classifiedData;
	
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
				
			if (this.bestGridPosition != null)
				this.model.setHyperParameterValues(this.bestGridPosition.getCoordinates(), this.trainData.getDatumTools());
		}
		long totalGridSearchTime = System.currentTimeMillis() - startGridSearch;
			
		output.debugWriteln("Training model with best parameters (" + this.name + ")");
		
		Pair<Long, Long> trainAndTestTime;
		
		evaluateConstraints(this.trainData, this.devData, this.testData);
		
		List<Double> evaluationValues = null;
		if (this.testData != null) {
			if (this.trainOnDev)
				this.trainData.addAll(this.devData);
	
			TrainTestValidation<D, L> accuracy = new TrainTestValidation<D, L>(this.name, this.model, this.trainData, this.testData, this.evaluations);
			evaluationValues = accuracy.run();
			if (evaluationValues.get(0) < 0) {
				output.debugWriteln("Error: Validation failed (" + this.name + ")");
				return null;
			} 
			
			this.classifiedData = accuracy.getClassifiedData();
			this.confusionMatrix = accuracy.getConfusionMatrix();
			output.debugWriteln("Test " + this.evaluations.get(0).toString() + " (" + this.name + ": " + cleanDouble.format(evaluationValues.get(0)));
			
			trainAndTestTime = accuracy.getTrainAndTestTime();
		} else {
			evaluationValues = this.bestGridPosition.getValidation().getResults();
			this.classifiedData = this.bestGridPosition.getValidation().getClassifiedData();
			this.confusionMatrix = this.bestGridPosition.getValidation().getConfusionMatrix();
			output.debugWriteln("Dev best " + this.evaluations.get(0).toString() + " (" + this.name + ": " + cleanDouble.format(evaluationValues.get(0)));
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
