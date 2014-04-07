package ark.model.evaluation;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.evaluation.metric.ClassificationEvaluation;
import ark.util.OutputWriter;
import ark.util.Pair;

public class GridSearch<D extends Datum<L>, L> {
	public static class GridPosition {
		private TreeMap<String, String> coordinates;
		
		public GridPosition() {
			this.coordinates = new TreeMap<String, String>();
		}
		
		public String getParameterValue(String parameter) {
			return this.coordinates.get(parameter);
		}
		
		public void setParameterValue(String parameter, String value) {
			this.coordinates.put(parameter, value);
		}
		
		public Map<String, String> getCoordinates() {
			return this.coordinates;
		}
		
		public GridPosition clone() {
			GridPosition clonePosition = new GridPosition();
			for (Entry<String, String> entry : this.coordinates.entrySet())
				clonePosition.setParameterValue(entry.getKey(), entry.getValue());
			return clonePosition;
		}
		
		public String toString() {
			StringBuilder str = new StringBuilder();
			str.append("(");
			for (Entry<String, String> entry : this.coordinates.entrySet()) {
				str.append(entry.getKey()).append("=").append(entry.getValue()).append(",");
			}
			str.delete(str.length() - 1, str.length());
			str.append(")");
			return str.toString();
		}
		
		public String toValueString(String separator) {
			StringBuilder str = new StringBuilder();
			for (Entry<String, String> entry : this.coordinates.entrySet()) {
				str.append(entry.getValue()).append(separator);
			}
			str.delete(str.length() - 1, str.length());
			return str.toString();
		}
		
		public String toKeyString(String separator) {
			StringBuilder str = new StringBuilder();
			for (Entry<String, String> entry : this.coordinates.entrySet()) {
				str.append(entry.getKey()).append(separator);
			}
			str.delete(str.length() - 1, str.length());
			return str.toString();
		}
		
		public String toKeyValueString(String separator, String keyValueGlue) {
			StringBuilder str = new StringBuilder();
			
			for (Entry<String, String> entry : this.coordinates.entrySet()) {
				str.append(entry.getKey())
				   .append(keyValueGlue)
				   .append(entry.getValue())
				   .append(separator);
			}
			
			str.delete(str.length() - 1, str.length());
			
			return str.toString();
		}
	}
	
	private String name;
	private SupervisedModel<D, L> model;
	private FeaturizedDataSet<D, L> trainData;
	private FeaturizedDataSet<D, L> testData;
	private Map<String, List<String>> possibleParameterValues;
	private List<Pair<GridPosition, Double>> gridEvaluation;
	private ClassificationEvaluation<D, L> evaluation;
	private DecimalFormat cleanDouble;
	
	public GridSearch(String name,
									SupervisedModel<D, L> model,
									FeaturizedDataSet<D, L> trainData, 
									FeaturizedDataSet<D, L> testData,
									Map<String, List<String>> possibleParameterValues,
									ClassificationEvaluation<D, L> evaluation) {
		this.name = name;
		this.model = model;
		this.trainData = trainData;
		this.testData = testData;
		this.possibleParameterValues = possibleParameterValues;
		this.gridEvaluation = null;
		this.evaluation = evaluation;
		this.cleanDouble = new DecimalFormat("0.00");
	}
	
	public String toString() {
		List<Pair<GridPosition, Double>> gridEvaluation = getGridEvaluation();
		StringBuilder gridEvaluationStr = new StringBuilder();
		
		gridEvaluationStr = gridEvaluationStr.append(gridEvaluation.get(0).getFirst().toKeyString("\t")).append("\t").append(this.evaluation.toString()).append("\n");
		for (Pair<GridPosition, Double> positionEvaluation : gridEvaluation) {
			gridEvaluationStr = gridEvaluationStr.append(positionEvaluation.getFirst().toValueString("\t"))
							 					 .append("\t")
							 					 .append(this.cleanDouble.format(positionEvaluation.getSecond()))
							 					 .append("\n");
		}
		
		return gridEvaluationStr.toString();
	}
	
	public List<Pair<GridPosition, Double>> getGridEvaluation() {
		if (this.gridEvaluation != null)
			return this.gridEvaluation;
		
		this.gridEvaluation = new ArrayList<Pair<GridPosition, Double>>();
		
		List<GridPosition> grid = constructGrid();
		for (GridPosition position : grid) {
			double positionValue = evaluateGridPosition(position);
			this.gridEvaluation.add(new Pair<GridPosition, Double>(position, positionValue));
		}
		
		return this.gridEvaluation;
	}
	
	public GridPosition getBestPosition() {
		List<Pair<GridPosition, Double>> gridEvaluation = getGridEvaluation();
		double maxValue = Double.NEGATIVE_INFINITY;
		GridPosition maxPosition = null;
		
		for (Pair<GridPosition, Double> positionValue : gridEvaluation) {
			if (positionValue.getSecond() > maxValue) {
				maxValue = positionValue.getSecond();
				maxPosition = positionValue.getFirst();
			}
		}
		
		return maxPosition;
	}
	
	private List<GridPosition> constructGrid() {
		List<GridPosition> positions = new ArrayList<GridPosition>();
		positions.add(new GridPosition());
		for (Entry<String, List<String>> possibleValuesEntry : this.possibleParameterValues.entrySet()) {
			List<GridPosition> newPositions = new ArrayList<GridPosition>();
			
			for (GridPosition position : positions) {
				for (String parameterValue : possibleValuesEntry.getValue()) {
					GridPosition newPosition = position.clone();
					newPosition.setParameterValue(possibleValuesEntry.getKey(), parameterValue);
					newPositions.add(newPosition);
				}
			}
			
			positions = newPositions;
		}
		
		return positions;
	}
	
	private double evaluateGridPosition(GridPosition position) {
		OutputWriter output = this.trainData.getDatumTools().getDataTools().getOutputWriter();
		
		output.debugWriteln("Grid search evaluating " + this.evaluation.toString() + " of model (" + this.name + " " + position.toString() + ")");
		
		SupervisedModel<D, L> positionModel = this.model.clone(this.trainData.getDatumTools());
		Map<String, String> parameterValues = position.getCoordinates();
		for (Entry<String, String> entry : parameterValues.entrySet()) {
			positionModel.setHyperParameterValue(entry.getKey(), entry.getValue(), this.trainData.getDatumTools());	
		}
		
		positionModel.setHyperParameterValue("warmRestart", "true", this.trainData.getDatumTools());
		
		List<ClassificationEvaluation<D, L>> evaluation = new ArrayList<ClassificationEvaluation<D, L>>(1);
		evaluation.add(this.evaluation);
		
		TrainTestValidation<D, L> validation = new TrainTestValidation<D, L>(this.name + " " + position.toString(), positionModel, this.trainData, this.testData, evaluation);
		double computedEvaluation = validation.run().get(0);
		if (computedEvaluation  < 0) {
			output.debugWriteln("Error: Grid search evaluation failed at position " + position.toString());
			return -1.0;
		}

		output.debugWriteln("Finished grid search evaluating model with hyper parameters (" + this.name + " " + position.toString() + ")");
		
		return computedEvaluation;
	}
}
