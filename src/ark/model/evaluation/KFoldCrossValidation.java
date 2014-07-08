package ark.model.evaluation;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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
 * KFoldCrossValidation performs a k-fold cross validation with 
 * a given model on set of annotated 
 * organization mentions 
 * (http://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation).  
 * For each fold, there is an  
 * optional grid-search for hyper-parameter values using
 * ark.model.evaluation.HyperParameterGridSearch with
 * (k-2) parts as training, one part as dev, and one part as
 * test data.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public class KFoldCrossValidation<D extends Datum<L>, L> {
	private String name;
	private SupervisedModel<D, L> model;
	private List<Feature<D, L>> features;
	private List<SupervisedModelEvaluation<D, L>> evaluations;
	private List<DataSet<D, L>> folds;
	// Map from hyper-parameters to possible values for grid search
	// This will be null if there shouldn't be a grid search
	private Map<String, List<String>> possibleParameterValues; 
	private DecimalFormat cleanDouble;
	
	/**
	 * @param name
	 * @param model
	 * @param features - Features (uninitialized) to use in the model
	 * @param evaluations - Metrics by which to evaluate the model.  The
	 * grid-search will use the first of these if there is a grid search.
	 * @param data - Dataset to randomly partition into k disjoint sets
	 * @param k - Number of folds
	 */
	public KFoldCrossValidation(String name,
								SupervisedModel<D, L> model, 
								List<Feature<D, L>> features,
								List<SupervisedModelEvaluation<D, L>> evaluations,
								DataSet<D, L> data,
								int k) {
		this.name = name;
		this.model = model;
		this.features = features;
		this.evaluations = evaluations;
		
		double[] foldDistribution = new double[k];
		for (int i = 0; i < k; i++)
			foldDistribution[i] = 1.0/k;

		this.folds = data.makePartition(foldDistribution, data.getDatumTools().getDataTools().getGlobalRandom());
		this.possibleParameterValues = new HashMap<String, List<String>>();
		this.cleanDouble = new DecimalFormat("0.00");
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
	
	public List<Double> run(int maxThreads, Datum.Tools.TokenSpanExtractor<D, L> errorExampleExtractor) {
		ConfusionMatrix<D, L> aggregateConfusions = new ConfusionMatrix<D, L>(this.model.getValidLabels(), this.model.getLabelMapping());
		
		/*
		 * Get results, using a separate thread for each fold
		 */
		ExecutorService threadPool = Executors.newFixedThreadPool(maxThreads);
		List<ValidationThread> tasks = new ArrayList<ValidationThread>();
		List<ValidationResult> validationResults = new ArrayList<ValidationResult>(this.folds.size());
 		for (int i = 0; i < this.folds.size(); i++) {
 			validationResults.add(null);
			tasks.add(new ValidationThread(i, 1, errorExampleExtractor));
		}
		
		try {
			List<Future<ValidationResult>> results = threadPool.invokeAll(tasks);
			threadPool.shutdown();
			threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
			for (Future<ValidationResult> futureResult : results) {
				ValidationResult result = futureResult.get();
				if (result == null)
					return null;
				validationResults.set(result.getFoldIndex(), result);
			}
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
		
		/* 
		 * Output results for each fold
		 */
		OutputWriter output = this.folds.get(0).getDatumTools().getDataTools().getOutputWriter();

		String gridSearchParameters = ((this.possibleParameterValues.size() > 0) ? validationResults.get(0).getBestParameters().toKeyString("\t") + "\t" : "");
		String evaluationsStr = "";
		List<Double> averageEvaluations = new ArrayList<Double>(this.evaluations.size());
		for (int i = 0; i < this.evaluations.size(); i++) {
			evaluationsStr += this.evaluations.get(i).toString() + "\t";
			averageEvaluations.add(0.0);
		}
		output.resultsWriteln("Fold\t" + gridSearchParameters + evaluationsStr);
		
		for (int i = 0; i < validationResults.size(); i++) {
			String gridSearchParameterValues = ((this.possibleParameterValues.size() > 0) ? validationResults.get(i).getBestParameters().toValueString("\t") + "\t" : "");
			List<Double> evaluationValues = validationResults.get(i).getEvaluationValues();
			String evaluationValuesStr = "";
			for (int j = 0; j < evaluationValues.size(); j++) {
				evaluationValuesStr += this.cleanDouble.format(evaluationValues.get(j)) + "\t";
				averageEvaluations.set(j , averageEvaluations.get(j) + evaluationValues.get(j));
			}
			
			output.resultsWriteln(i + "\t" + gridSearchParameterValues + evaluationValuesStr);
			aggregateConfusions.add(validationResults.get(i).getConfusionMatrix());
		}
		
		output.resultsWrite("Averages:\t");
		for (int i = 0; i < this.possibleParameterValues.size(); i++)
			output.resultsWrite("\t");
		for (int i = 0; i < averageEvaluations.size(); i++) {
			averageEvaluations.set(i, averageEvaluations.get(i)/this.folds.size());
			output.resultsWrite(this.cleanDouble.format(averageEvaluations.get(i)) + "\t");
		}
		output.resultsWriteln("");
		
		/*
		 * Output confusion matrix
		 */
		output.resultsWriteln("\nTotal Confusion Matrix:\n " + aggregateConfusions.toString());
		
		/*
		 * Output grid search results
		 */
		if (this.possibleParameterValues.size() > 0) {
			output.resultsWriteln("\nGrid search results:");
			output.resultsWrite(validationResults.get(0).getGridEvaluation().get(0).toKeyString("\t") + "\t");
			for (int i = 0; i < this.folds.size(); i++)
				output.resultsWrite("Fold " + i + "\t");
			output.resultsWrite("\n");
			
			List<Pair<GridSearch<D, L>.GridPosition, List<Double>>> gridFoldResults = new ArrayList<Pair<GridSearch<D, L>.GridPosition, List<Double>>>();
			for (int i = 0; i < validationResults.size(); i++) {
				List<GridSearch<D, L>.EvaluatedGridPosition> gridEvaluation = validationResults.get(i).getGridEvaluation();
				for (int j = 0; j < gridEvaluation.size(); j++) {
					if (gridFoldResults.size() <= j)
						gridFoldResults.add(new Pair<GridSearch<D, L>.GridPosition, List<Double>>(gridEvaluation.get(j), new ArrayList<Double>()));
					gridFoldResults.get(j).getSecond().add(gridEvaluation.get(j).getPositionValue());
				}
			}
			
			for (Pair<GridSearch<D, L>.GridPosition, List<Double>> gridFoldResult : gridFoldResults) {
				output.resultsWrite(gridFoldResult.getFirst().toValueString("\t") + "\t");
				for (int i = 0; i < gridFoldResult.getSecond().size(); i++) {
					output.resultsWrite(this.cleanDouble.format(gridFoldResult.getSecond().get(i)) + "\t");
				}
				output.resultsWrite("\n");
			}
		}
		
		return averageEvaluations;
	}
	
	/**
	 * ValidationResult stores the results of training and evaluating 
	 * the model on a single fold
	 * 
	 * @author Bill McDowell
	 *
	 */
	private class ValidationResult  {
		private int foldIndex;
		private List<Double> evaluationValues;
		private ConfusionMatrix<D, L> confusionMatrix;
		private List<GridSearch<D, L>.EvaluatedGridPosition> gridEvaluation;
		private GridSearch<D, L>.GridPosition bestParameters;
		
		public ValidationResult(int foldIndex, List<Double> evaluationValues, ConfusionMatrix<D, L> confusionMatrix, List<GridSearch<D, L>.EvaluatedGridPosition> gridEvaluation, GridSearch<D, L>.GridPosition bestParameters) {
			this.foldIndex = foldIndex;
			this.evaluationValues = evaluationValues;
			this.confusionMatrix = confusionMatrix;
			this.gridEvaluation = gridEvaluation;
			this.bestParameters = bestParameters;
		}
		
		public int getFoldIndex() {
			return this.foldIndex;
		}
		
		public List<Double> getEvaluationValues() {
			return this.evaluationValues;
		}
		
		public ConfusionMatrix<D, L> getConfusionMatrix() {
			return this.confusionMatrix;
		}
		
		public List<GridSearch<D, L>.EvaluatedGridPosition> getGridEvaluation() {
			return this.gridEvaluation;
		}
		
		public GridSearch<D, L>.GridPosition getBestParameters() {
			return this.bestParameters;
		}
	}
	
	/**
	 * ValidationThread trains and evaluates the model on a single
	 * fold (with optional single-threaded grid search).
	 * 
	 * @author Bill McDowell
	 *
	 */
	private class ValidationThread implements Callable<ValidationResult> {
		private int foldIndex;
		private int maxThreads;
		private Datum.Tools.TokenSpanExtractor<D, L> errorExampleExtractor;
		// fold-specific environment variables that can be referenced by
		// experiment configuration files
		private Map<String, String> parameterEnvironment; 
		
		public ValidationThread(int foldIndex, int maxThreads, Datum.Tools.TokenSpanExtractor<D, L> errorExampleExtractor) {
			this.foldIndex = foldIndex;
			this.maxThreads = maxThreads;
			this.errorExampleExtractor = errorExampleExtractor;
	
			this.parameterEnvironment = new HashMap<String, String>();
			this.parameterEnvironment.putAll(folds.get(foldIndex).getDatumTools().getDataTools().getParameterEnvironment());
			this.parameterEnvironment.put("FOLD", String.valueOf(this.foldIndex));
		}
		
		public ValidationResult call() {
			OutputWriter output = folds.get(foldIndex).getDatumTools().getDataTools().getOutputWriter();
			String namePrefix = name + " Fold " + foldIndex;
			
			/*
			 * Initialize training, dev, and test sets
			 */
			output.debugWriteln("Initializing CV data sets for " + name);
			Datum.Tools<D, L> datumTools = folds.get(this.foldIndex).getDatumTools();
			Datum.Tools.LabelMapping<L> labelMapping = folds.get(this.foldIndex).getLabelMapping();
			FeaturizedDataSet<D, L> testData = new FeaturizedDataSet<D, L>(namePrefix + " Test", features, this.maxThreads, datumTools, labelMapping);
			FeaturizedDataSet<D, L> trainData = new FeaturizedDataSet<D, L>(namePrefix + " Training", features, this.maxThreads, datumTools, labelMapping);
			FeaturizedDataSet<D, L> devData = new FeaturizedDataSet<D, L>(namePrefix + " Dev", features, this.maxThreads, datumTools, labelMapping);
			for (int j = 0; j < folds.size(); j++) {
				if (j == this.foldIndex) {
					testData.addAll(folds.get(j));
				} else if (possibleParameterValues.size() > 0 && j == ((foldIndex + 1) % folds.size())) {
					devData.addAll(folds.get(j));
				} else {
					trainData.addAll(folds.get(j));	
				}
			}
			
			/* Need cloned bunch of features for the fold so that they can be 
			 * reinitialized without affecting other folds' results */
			output.debugWriteln("Initializing features for CV fold " + this.foldIndex);
			for (Feature<D, L> feature : features) {
				Feature<D, L> foldFeature = feature.clone(datumTools, this.parameterEnvironment);
				if (!foldFeature.init(trainData))
					return null;
				
				trainData.addFeature(foldFeature);
				devData.addFeature(foldFeature);
				testData.addFeature(foldFeature);
			}
			
			SupervisedModel<D, L> foldModel = model.clone(datumTools, this.parameterEnvironment);
			
			output.dataWriteln("--------------- Fold: " + this.foldIndex + " ---------------");
			output.modelWriteln("--------------- Fold: " + this.foldIndex + " ---------------");
			
			/*
			 *  Run either TrainTestValidation or GridSearchTestValidation on the fold
			 */
			ValidationResult result = null;
			List<Double> evaluationValues = null;
			if (possibleParameterValues.size() > 0) {
				GridSearchTestValidation<D, L> gridSearchValidation = new GridSearchTestValidation<D, L>(namePrefix, foldModel, trainData, devData, testData, evaluations, true);
				gridSearchValidation.setPossibleHyperParameterValues(possibleParameterValues);
				evaluationValues = gridSearchValidation.run(this.errorExampleExtractor, false);
				result = new ValidationResult(foldIndex, evaluationValues, gridSearchValidation.getConfusionMatrix(), gridSearchValidation.getGridEvaluation(), gridSearchValidation.getBestGridPosition());
			} else {
				TrainTestValidation<D, L> accuracyValidation = new TrainTestValidation<D, L>(namePrefix, foldModel, trainData, testData, evaluations);
				evaluationValues = accuracyValidation.run();
				output.modelWriteln(accuracyValidation.getModel().toString());
				result = new ValidationResult(foldIndex, evaluationValues, accuracyValidation.getConfusionMatrix(), null, null);
			}
			
			if (evaluationValues.get(0) < 0)
				output.debugWriteln("Error: Validation failed on fold " + this.foldIndex);
			
			return result;
		}
	}
}