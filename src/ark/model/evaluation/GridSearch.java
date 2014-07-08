package ark.model.evaluation;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.Map.Entry;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.OutputWriter;
import ark.util.Pair;

/**
 * GridSearch performs a grid-search for hyper-parameter values
 * of a given model using a training and test (dev) data
 * set.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public class GridSearch<D extends Datum<L>, L> {
	/**
	 * GridPosition represents a position in the grid
	 * of parameter values (a setting of values for the
	 * parameters)
	 * 
	 * @author Bill McDowell
	 *
	 */
	public class GridPosition {
		protected TreeMap<String, String> coordinates;
		
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
	
	/**
	 * EvaluatedGridPosition is a GridPosition that has been
	 * evaluated according to some measure and given a value
	 * 
	 * @author Bill McDowell
	 *
	 */
	public class EvaluatedGridPosition extends GridPosition {
		private double positionValue;
		private TrainTestValidation<D, L> validation; // determines the evaluation
		private Pair<Long, Long> trainAndTestTime;
		
		public EvaluatedGridPosition(GridPosition position, double positionValue, TrainTestValidation<D, L> validation, Pair<Long, Long> trainAndTestTime) {
			this.coordinates = position.coordinates;
			this.positionValue = positionValue;
			this.validation = validation;
			this.trainAndTestTime = trainAndTestTime;
		}

		
		public double getPositionValue() {
			return this.positionValue;
		}
		
		public TrainTestValidation<D, L> getValidation() {
			return this.validation;
		}
		
		public Pair<Long, Long> getTrainAndTestTime(){
			return this.trainAndTestTime;
		}
	}
	
	private String name;
	private SupervisedModel<D, L> model;
	private FeaturizedDataSet<D, L> trainData;
	private FeaturizedDataSet<D, L> testData;
	// map from parameters to lists of possible values
	private Map<String, List<String>> possibleParameterValues;
	// grid of evaluated grid positions (evaluated parameter settings)
	private List<EvaluatedGridPosition> gridEvaluation;
	// evaluation measure
	private SupervisedModelEvaluation<D, L> evaluation;
	private DecimalFormat cleanDouble;
	
	/**
	 * 
	 * @param name
	 * @param model
	 * @param trainData
	 * @param testData
	 * @param possibleParameterValues - Map from parameters to lists of their possible values
	 * @param evaluation - Evaluation measure by which to search
	 */
	public GridSearch(String name,
									SupervisedModel<D, L> model,
									FeaturizedDataSet<D, L> trainData, 
									FeaturizedDataSet<D, L> testData,
									Map<String, List<String>> possibleParameterValues,
									SupervisedModelEvaluation<D, L> evaluation) {
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
		List<EvaluatedGridPosition> gridEvaluation = getGridEvaluation();
		StringBuilder gridEvaluationStr = new StringBuilder();
		
		gridEvaluationStr = gridEvaluationStr.append(gridEvaluation.get(0).toKeyString("\t")).append("\t").append(this.evaluation.toString()).append("\n");
		for (EvaluatedGridPosition positionEvaluation : gridEvaluation) {
			gridEvaluationStr = gridEvaluationStr.append(positionEvaluation.toValueString("\t"))
							 					 .append("\t")
							 					 .append(this.cleanDouble.format(positionEvaluation.getPositionValue()))
							 					 .append("\n");
		}
		
		return gridEvaluationStr.toString();
	}
	
	public List<EvaluatedGridPosition> getGridEvaluation() {
		return getGridEvaluation(1);
	}
	
	public List<EvaluatedGridPosition> getGridEvaluation(int maxThreads) {
		if (this.gridEvaluation != null)
			return this.gridEvaluation;
		
		this.gridEvaluation = new ArrayList<EvaluatedGridPosition>();
		List<GridPosition> grid = constructGrid();
		
		ExecutorService threadPool = Executors.newFixedThreadPool(maxThreads);
		List<PositionThread> tasks = new ArrayList<PositionThread>();
 		for (GridPosition position : grid) {
			tasks.add(new PositionThread(position));
		}
		
		try {
			List<Future<EvaluatedGridPosition>> results = threadPool.invokeAll(tasks);
			threadPool.shutdown();
			threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
			for (Future<EvaluatedGridPosition> futureResult : results) {
				EvaluatedGridPosition result = futureResult.get();
				if (result == null)
					return null;
				this.gridEvaluation.add(result);
			}
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
		
		return this.gridEvaluation;
	}
	
	public EvaluatedGridPosition getBestPosition() {
		return getBestPosition(1);
	}
	
	public EvaluatedGridPosition getBestPosition(int maxThreads) {
		List<EvaluatedGridPosition> gridEvaluation = getGridEvaluation(maxThreads);
		double maxValue = Double.NEGATIVE_INFINITY;
		EvaluatedGridPosition maxPosition = null;
		
		for (EvaluatedGridPosition positionValue : gridEvaluation) {
			if (positionValue.getPositionValue() > maxValue) {
				maxValue = positionValue.getPositionValue();
				maxPosition = positionValue;
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
	
	private class PositionThread implements Callable<EvaluatedGridPosition> {
		private GridPosition position;
		
		public PositionThread(GridPosition position) {
			this.position = position;
		}
		
		@Override
		public EvaluatedGridPosition call() throws Exception {
			OutputWriter output = trainData.getDatumTools().getDataTools().getOutputWriter();
			
			output.debugWriteln("Grid search evaluating " + evaluation.toString() + " of model (" + name + " " + position.toString() + ")");
			
			SupervisedModel<D, L> positionModel = model.clone(trainData.getDatumTools());
			Map<String, String> parameterValues = position.getCoordinates();
			for (Entry<String, String> entry : parameterValues.entrySet()) {
				positionModel.setHyperParameterValue(entry.getKey(), entry.getValue(), trainData.getDatumTools());	
			}
			
			//positionModel.setHyperParameterValue("warmRestart", "true", this.trainData.getDatumTools());
			
			List<SupervisedModelEvaluation<D, L>> evaluations = new ArrayList<SupervisedModelEvaluation<D, L>>(1);
			evaluations.add(evaluation);
			
			TrainTestValidation<D, L> validation = new TrainTestValidation<D, L>(name + " " + position.toString(), positionModel, trainData, testData, evaluations);
			double computedEvaluation = validation.run().get(0);
			if (computedEvaluation  < 0) {
				output.debugWriteln("Error: Grid search evaluation failed at position " + position.toString());
				return null;
			}

			output.debugWriteln("Finished grid search evaluating model with hyper parameters (" + name + " " + position.toString() + ")");
			
			return new EvaluatedGridPosition(this.position, computedEvaluation, validation, validation.getTrainAndTestTime());
		}
		
	}
}
