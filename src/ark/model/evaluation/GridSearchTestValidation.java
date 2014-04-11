package ark.model.evaluation;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.evaluation.metric.ClassificationEvaluation;
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
	private List<ClassificationEvaluation<D, L>> evaluations;
	private DecimalFormat cleanDouble;
	private List<Pair<GridSearch.GridPosition, Double>> gridEvaluation;
	private GridSearch.GridPosition bestGridPosition;
	private boolean trainOnDev;
	
	public GridSearchTestValidation(String name,
							  SupervisedModel<D, L> model, 
							  FeaturizedDataSet<D, L> trainData,
							  FeaturizedDataSet<D, L> devData,
							  FeaturizedDataSet<D, L> testData,
							  List<ClassificationEvaluation<D, L>> evaluations,
							  boolean trainOnDev) {
		this.name = name;
		this.model = model;
		this.trainData = trainData;
		this.devData = devData;
		this.testData = testData;
		this.evaluations = evaluations;
		this.possibleParameterValues = new HashMap<String, List<String>>();
		this.gridEvaluation = new ArrayList<Pair<GridSearch.GridPosition, Double>>();
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
		OutputWriter output = this.trainData.getDatumTools().getDataTools().getOutputWriter();
		
		if (this.possibleParameterValues.size() > 0) {
			GridSearch<D, L> gridSearch = new GridSearch<D,L>(this.name,
										this.model,
						 				this.trainData, 
						 				this.devData,
						 				this.possibleParameterValues,
						 				this.evaluations.get(0)); 
			this.bestGridPosition = gridSearch.getBestPosition();
			this.gridEvaluation = gridSearch.getGridEvaluation();
				
			output.debugWriteln("Grid search (" + this.name + ": \n" + gridSearch.toString());
				
			if (this.bestGridPosition != null)
				this.model.setHyperParameterValues(this.bestGridPosition.getCoordinates(), this.trainData.getDatumTools());
		}

		if (this.trainOnDev)
			this.trainData.addAll(this.devData);
			
		output.debugWriteln("Training model with best parameters (" + this.name + ")");

		TrainTestValidation<D, L> accuracy = new TrainTestValidation<D, L>(this.name, this.model, this.trainData, this.testData, this.evaluations);
		List<Double> evaluationValues = accuracy.run();
		if (evaluationValues.get(0) < 0) {
			output.debugWriteln("Error: Validation failed (" + this.name + ")");
			return null;
		} else {
			this.confusionMatrix = accuracy.getConfusionMatrix();
			
			output.debugWriteln("Test " + this.evaluations.get(0).toString() + " (" + this.name + ": " + cleanDouble.format(evaluationValues.get(0)));
			output.dataWriteln(this.confusionMatrix.getActualToPredictedDescription(errorExampleExtractor));
			output.modelWriteln(this.model.toString());
			
			if (outputResults) {
				if (this.bestGridPosition != null) {
					Map<String, String> parameters = this.bestGridPosition.getCoordinates();
					output.resultsWriteln("Best parameters from grid search:");
					for (Entry<String, String> entry : parameters.entrySet())
						output.resultsWriteln(entry.getKey() + ": " + entry.getValue());
				}
				
				output.resultsWriteln("\nTest set evaluation results:");
				for (int i = 0; i < this.evaluations.size(); i++)
					output.resultsWriteln(this.evaluations.get(i).toString() + ": " + evaluationValues.get(i));
				
				output.resultsWriteln("\nConfusion matrix:\n" + this.confusionMatrix.toString());
				
				if (this.gridEvaluation != null && this.gridEvaluation.size() > 0) {
					output.resultsWriteln("\nGrid search on " + this.evaluations.get(0).toString() + ":");
					output.resultsWriteln(this.gridEvaluation.get(0).getFirst().toKeyString("\t") + "\t" + this.evaluations.get(0).toString());
					for (Pair<GridSearch.GridPosition, Double> gridPosition : this.gridEvaluation) {
						output.resultsWriteln(gridPosition.getFirst().toValueString("\t") + "\t" + gridPosition.getSecond());
					}
				}
			}
			
			return evaluationValues;
		}
	}
	
	public List<ClassificationEvaluation<D, L>> getEvaluations() {
		return this.evaluations;
	}
	
	public ConfusionMatrix<D, L> getConfusionMatrix() {
		return this.confusionMatrix;
	}
	
	public List<Pair<GridSearch.GridPosition, Double>> getGridEvaluation() {
		return this.gridEvaluation;
	}
	
	public GridSearch.GridPosition getBestGridPosition() {
		return this.bestGridPosition;
	}
}
