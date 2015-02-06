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

import java.io.BufferedReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
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
import ark.util.SerializationUtil;

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
	public static class GridDimension {
		private String name;
		private List<String> values;
		private boolean trainingDimension;
		
		public GridDimension() {
			this.values = new ArrayList<String>();
			this.trainingDimension = true;
		}
		
		public boolean deserialize(BufferedReader reader) throws IOException {
			this.name = SerializationUtil.deserializeGenericName(reader);
			boolean hasDimensionValues = false;
			Map<String, String> parameters = SerializationUtil.deserializeArguments(reader);
			for (Entry<String, String> parameter : parameters.entrySet()) {
				if (parameter.getKey().equals("values")) {
					String[] values = parameter.getValue().split(",");
					for (String value : values) {
						this.values.add(value.trim());
					}
					
					hasDimensionValues =  true;
				} else if (parameter.getKey().equals("training")) {
					this.trainingDimension = Boolean.valueOf(parameter.getValue());
				}
			}
			
			return hasDimensionValues;
		}
		
		public boolean isTrainingDimension() {
			return this.trainingDimension;
		}
		
		public List<String> getValues() {
			return this.values;
		}
		
		public String getName() {
			return this.name;
		}
	}
	
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
		
		@SuppressWarnings("unchecked")
		@Override
		public boolean equals(Object o) {
			GridPosition g = (GridPosition)o;
			
			if (g.coordinates.size() != this.coordinates.size())
				return false;
			
			for (Entry<String, String> entry : this.coordinates.entrySet())
				if (!g.coordinates.containsKey(entry.getKey()) || !g.coordinates.get(entry.getKey()).equals(entry.getValue()))
					return false;
			
			return true;
		}
		
		@Override
		public int hashCode() {
			int hashCode = 0;
			
			for (Entry<String, String> entry : this.coordinates.entrySet())
				hashCode ^= entry.getKey().hashCode() ^ entry.getValue().hashCode();
			
			return hashCode;
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
		private ValidationTrainTest<D, L> validation;
		
		public EvaluatedGridPosition(GridPosition position, double positionValue, ValidationTrainTest<D, L> validation) {
			this.coordinates = position.coordinates;
			this.positionValue = positionValue;
			this.validation = validation;
		}

		
		public double getPositionValue() {
			return this.positionValue;
		}
		
		public ValidationTrainTest<D, L> getValidation() {
			return this.validation;
		}
	}
	
	private String name;
	private SupervisedModel<D, L> model;
	private FeaturizedDataSet<D, L> trainData;
	private FeaturizedDataSet<D, L> testData;
	
	private List<GridDimension> dimensions;
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
	 * @param dimensions - Grid dimensions and their possible values
	 * @param evaluation - Evaluation measure by which to search
	 */
	public GridSearch(String name,
									SupervisedModel<D, L> model,
									FeaturizedDataSet<D, L> trainData, 
									FeaturizedDataSet<D, L> testData,
									List<GridDimension> dimensions,
									SupervisedModelEvaluation<D, L> evaluation) {
		this.name = name;
		this.model = model;
		this.trainData = trainData;
		this.testData = testData;
		this.dimensions = dimensions;
		this.gridEvaluation = null;
		this.evaluation = evaluation;
		this.cleanDouble = new DecimalFormat("0.00000");
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
			List<Future<List<EvaluatedGridPosition>>> results = threadPool.invokeAll(tasks);
			threadPool.shutdown();
			threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
			for (Future<List<EvaluatedGridPosition>> futureResult : results) {
				List<EvaluatedGridPosition> result = futureResult.get();
				if (result == null)
					return null;
				this.gridEvaluation.addAll(result);
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
		
		for (EvaluatedGridPosition position: gridEvaluation) {
			if (position.getPositionValue() > maxValue) {
				maxValue = position.getPositionValue();
				maxPosition = position;
			}
		}
		
		// FIXME This is a hack.  Currently non-training parameters will be same across
		// all resulting models since models aren't cloned for non-training parameters
		maxPosition.getValidation().getModel().setHyperParameterValues(maxPosition.coordinates, maxPosition.getValidation().datumTools);
		
		return maxPosition;
	}
	
	private List<GridPosition> constructGrid() {
		return constructGrid(null, true);
	}
	
	private List<GridPosition> constructGrid(GridPosition initialPosition, boolean training) {
		List<GridPosition> positions = new ArrayList<GridPosition>();
		
		if (initialPosition != null) 
			positions.add(initialPosition);
		else
			positions.add(new GridPosition());
		
		for (GridDimension dimension : this.dimensions) {
			if (dimension.isTrainingDimension() != training)
				continue;
			
			List<GridPosition> newPositions = new ArrayList<GridPosition>();
			
			for (GridPosition position : positions) {
				for (String value : dimension.getValues()) {
					GridPosition newPosition = position.clone();
					newPosition.setParameterValue(dimension.getName(), value);
					newPositions.add(newPosition);
				}
			}
			
			positions = newPositions;
		}
		
		return positions;
	}
	
	private class PositionThread implements Callable<List<EvaluatedGridPosition>> {
		private GridPosition position;
		private Map<String, String> parameterEnvironment;
		private SupervisedModel<D, L> positionModel;
		
		public PositionThread(GridPosition position) {
			this.position = position;
			this.parameterEnvironment = new HashMap<String, String>();
			this.parameterEnvironment.putAll(trainData.getDatumTools().getDataTools().getParameterEnvironment());
			this.parameterEnvironment.putAll(this.position.getCoordinates());
			this.positionModel = model.clone(trainData.getDatumTools(), this.parameterEnvironment, true);
		}
		
		@Override
		public List<EvaluatedGridPosition> call() throws Exception {
			List<GridPosition> positions = constructGrid(this.position, false); // Positions for non-training dimensions
			List<EvaluatedGridPosition> evaluatedPositions = new ArrayList<EvaluatedGridPosition>();
			boolean skipTraining = false;
			for (GridPosition position : positions) {
				evaluatedPositions.add(evaluatePosition(position, skipTraining));
				skipTraining = true;
			}
			
			return evaluatedPositions;
		}
		
		private EvaluatedGridPosition evaluatePosition(GridPosition position, boolean skipTraining) {
			OutputWriter output = trainData.getDatumTools().getDataTools().getOutputWriter();
			
			output.debugWriteln("Grid search evaluating " + evaluation.toString() + " of model (" + name + " " + position.toString() + ")");
			
			Map<String, String> parameterValues = position.getCoordinates();
			for (Entry<String, String> entry : parameterValues.entrySet()) {
				this.positionModel.setParameterValue(entry.getKey(), entry.getValue(), trainData.getDatumTools());	
			}
			
			List<SupervisedModelEvaluation<D, L>> evaluations = new ArrayList<SupervisedModelEvaluation<D, L>>(1);
			evaluations.add(evaluation);
			
			ValidationTrainTest<D, L> validation = new ValidationTrainTest<D, L>(name + " " + position.toString(), 1, this.positionModel, trainData, testData, evaluations, null);
			double computedEvaluation = validation.run(skipTraining).get(0);
			if (computedEvaluation  < 0) {
				output.debugWriteln("Error: Grid search evaluation failed at position " + position.toString());
				return null;
			}
			
			output.debugWriteln("Finished grid search evaluating model with hyper parameters (" + name + " " + position.toString() + ")");
			
			return new EvaluatedGridPosition(position, computedEvaluation, validation);
		}
	}
}
