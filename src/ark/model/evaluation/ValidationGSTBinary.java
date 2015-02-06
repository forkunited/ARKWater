package ark.model.evaluation;

import java.io.BufferedReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.annotation.Datum.Tools.LabelIndicator;
import ark.data.annotation.Datum.Tools.TokenSpanExtractor;
import ark.data.feature.Feature;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.model.SupervisedModelCompositeBinary;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.OutputWriter;
import ark.util.SerializationUtil;
import ark.util.ThreadMapper;
import ark.util.ThreadMapper.Fn;
import ark.util.Timer;

/**
 * ValidationGSTBinary performs ValidationGST validations on several
 * binary models in parallel.  There is a separate validation for
 * each label indicator stored in the given Datum.Tools object.
 * 
 * @author Bill McDowell
 *
 * @param <T>
 * @param <D>
 * @param <L>
 */
public class ValidationGSTBinary<T extends Datum<Boolean>, D extends Datum<L>, L> extends ValidationGST<D, L> {
	private ConfusionMatrix<T, Boolean> aggregateConfusionMatrix;
	private List<ValidationGST<T, Boolean>> binaryValidations;
	private Map<GridSearch<T, Boolean>.GridPosition, Integer> bestPositionCounts;
	private SupervisedModelCompositeBinary<T, D, L> learnedCompositeModel;
	
	private List<SupervisedModelEvaluation<D, L>> compositeEvaluations;
	private Datum.Tools.InverseLabelIndicator<L> inverseLabelIndicator;
	private List<Double> compositeEvaluationValues;
	
	public ValidationGSTBinary(String name,
			  int maxThreads,
			  SupervisedModel<D, L> model, 
			  FeaturizedDataSet<D, L> trainData,
			  FeaturizedDataSet<D, L> devData,
			  FeaturizedDataSet<D, L> testData,
			  List<SupervisedModelEvaluation<D, L>> evaluations,
			  TokenSpanExtractor<D, L> errorExampleExtractor,
			  List<GridSearch.GridDimension> gridDimensions,
			  boolean trainOnDev,
			  List<SupervisedModelEvaluation<D, L>> compositeEvaluations,
			  Datum.Tools.InverseLabelIndicator<L> inverseLabelIndicator) {
		super(name, 
			  maxThreads,
			  model,
			  trainData,
			  devData,
			  testData,
			  evaluations,
			  errorExampleExtractor,
			  gridDimensions,
			  trainOnDev);
	
		this.inverseLabelIndicator = inverseLabelIndicator;
		this.compositeEvaluations = compositeEvaluations;
		this.compositeEvaluationValues = new ArrayList<Double>();
	}

	public ValidationGSTBinary(String name,
							   int maxThreads,
							   SupervisedModel<D, L> model, 
							   List<Feature<D, L>> features,
							   DataSet<D, L> trainData,
							   DataSet<D, L> devData,
							   DataSet<D, L> testData,
							   List<SupervisedModelEvaluation<D, L>> evaluations,
							   TokenSpanExtractor<D, L> errorExampleExtractor,
							   List<GridSearch.GridDimension> gridDimensions,
							   boolean trainOnDev,
							   Datum.Tools.InverseLabelIndicator<L> inverseLabelIndicator,
							   List<SupervisedModelEvaluation<D, L>> compositeEvaluations) {
		super(name, 
			  maxThreads,
			  model,
			  features,
			  trainData,
			  devData,
			  testData,
			  evaluations,
			  errorExampleExtractor,
			  gridDimensions,
			  trainOnDev);
		
		this.inverseLabelIndicator = inverseLabelIndicator;
		this.compositeEvaluations = compositeEvaluations;
		this.compositeEvaluationValues = new ArrayList<Double>();
	}

	public ValidationGSTBinary(String name, 
				 Datum.Tools<D, L> datumTools, 
				 DataSet<D, L> trainData, 
				 DataSet<D, L> devData, 
				 DataSet<D, L> testData,
				 Datum.Tools.InverseLabelIndicator<L> inverseLabelIndicator) {
		super(name, datumTools, trainData, devData, testData);
		this.inverseLabelIndicator = inverseLabelIndicator;
		this.compositeEvaluations = new ArrayList<SupervisedModelEvaluation<D, L>>();
		this.compositeEvaluationValues = new ArrayList<Double>();
	}

	public ValidationGSTBinary(String name, 
						 Datum.Tools<D, L> datumTools, 
						 FeaturizedDataSet<D, L> trainData, 
						 FeaturizedDataSet<D, L> devData, 
						 FeaturizedDataSet<D, L> testData,
						 Datum.Tools.InverseLabelIndicator<L> inverseLabelIndicator) {
		super(name, datumTools, trainData, devData, testData);
		this.inverseLabelIndicator = inverseLabelIndicator;
		this.compositeEvaluations = new ArrayList<SupervisedModelEvaluation<D, L>>();
		this.compositeEvaluationValues = new ArrayList<Double>();
	}
	
	public ValidationGSTBinary(String name, Datum.Tools<D, L> datumTools, Datum.Tools.InverseLabelIndicator<L> inverseLabelIndicator) {
		super(name, datumTools);
		this.inverseLabelIndicator = inverseLabelIndicator;
		this.compositeEvaluations = new ArrayList<SupervisedModelEvaluation<D, L>>();
		this.compositeEvaluationValues = new ArrayList<Double>();
	}
	
	@Override
	public boolean reset(FeaturizedDataSet<D, L> trainData, FeaturizedDataSet<D, L> devData, FeaturizedDataSet<D, L> testData) {
		if (!super.reset(trainData, devData, testData))
			return false;
	
		this.aggregateConfusionMatrix = null;
		this.binaryValidations = null;
		this.bestPositionCounts = null;
		this.learnedCompositeModel = null;
		this.compositeEvaluationValues = new ArrayList<Double>();
		
		return true;
	}
	
	@Override 
	public SupervisedModel<D, L> getModel() {
		return this.learnedCompositeModel;
	}
	
	@Override
	public List<Double> run() { 
		Timer timer = this.trainData.getDatumTools().getDataTools().getTimer();
		final OutputWriter output = this.trainData.getDatumTools().getDataTools().getOutputWriter();
		final Set<Boolean> binaryLabels = new HashSet<Boolean>();
		binaryLabels.add(true);
		binaryLabels.add(false);
		
		if (!this.trainData.getPrecomputedFeatures() 
				|| !this.devData.getPrecomputedFeatures() 
				|| (this.testData != null && !this.testData.getPrecomputedFeatures())) {
			timer.startClock(this.name + " Feature Computation");
			output.debugWriteln("Computing features...");
			if (!this.trainData.precomputeFeatures()
					|| !this.devData.precomputeFeatures()
					|| (this.testData != null && !this.testData.precomputeFeatures()))
				return null;
			output.debugWriteln("Finished computing features.");
			timer.stopClock(this.name + " Feature Computation");
		}
		output.resultsWriteln("Training data examples: " + this.trainData.size());
		output.resultsWriteln("Dev data examples: " + this.devData.size());
		output.resultsWriteln("Test data examples: " + this.testData.size());
		output.resultsWriteln("Feature vocabulary size: " + this.trainData.getFeatureVocabularySize() + "\n");
		
		ThreadMapper<LabelIndicator<L>, ValidationGST<T, Boolean>> threads = new ThreadMapper<LabelIndicator<L>, ValidationGST<T, Boolean>>(new Fn<LabelIndicator<L>, ValidationGST<T, Boolean>>() {
			public ValidationGST<T, Boolean> apply(LabelIndicator<L> labelIndicator) {
				Datum.Tools<T, Boolean> binaryTools = trainData.getDatumTools().makeBinaryDatumTools(labelIndicator);
				SupervisedModel<T, Boolean> binaryModel = model.clone(binaryTools, binaryTools.getDataTools().getParameterEnvironment(), false);
				
				binaryModel.setLabels(binaryLabels, null);
				
				List<SupervisedModelEvaluation<T, Boolean>> binaryEvaluations = new ArrayList<SupervisedModelEvaluation<T, Boolean>>();
				for (SupervisedModelEvaluation<D, L> evaluation : evaluations) {
					binaryEvaluations.add(evaluation.clone(binaryTools, binaryTools.getDataTools().getParameterEnvironment(), false));
				}
				
				FeaturizedDataSet<T, Boolean> binaryTrainData = (FeaturizedDataSet<T, Boolean>)trainData.makeBinaryDataSet(labelIndicator.toString(), binaryTools);
				FeaturizedDataSet<T, Boolean> binaryDevData = (FeaturizedDataSet<T, Boolean>)devData.makeBinaryDataSet(labelIndicator.toString(), binaryTools);
				FeaturizedDataSet<T, Boolean> binaryTestData = (testData == null) ? null : (FeaturizedDataSet<T, Boolean>)testData.makeBinaryDataSet(labelIndicator.toString(), binaryTools);
				
				ValidationGST<T, Boolean> binaryValidation = new ValidationGST<T, Boolean>(
						labelIndicator.toString(),
						(int)Math.ceil(maxThreads / (double)trainData.getDatumTools().getLabelIndicators().size()),
						binaryModel,
						binaryTrainData,
						binaryDevData,
						binaryTestData,			
						binaryEvaluations,
						binaryTools.getTokenSpanExtractor(errorExampleExtractor.toString()),
						gridDimensions,
						trainOnDev);
				
				if (binaryTrainData.getDataSizeForLabel(true) == 0
						|| binaryDevData.getDataSizeForLabel(true) == 0 
						|| (binaryTestData != null && binaryTestData.getDataSizeForLabel(true) == 0)) {
					output.debugWriteln("Skipping " + labelIndicator.toString() + ".  Not enough positive examples.");
					return binaryValidation;
				}
					
				if (!binaryValidation.runAndOutput()) {
					return null;
				}
				
				return binaryValidation;
			}
		});
		
		this.binaryValidations = threads.run(this.trainData.getDatumTools().getLabelIndicators(), this.maxThreads);
		this.aggregateConfusionMatrix = new ConfusionMatrix<T, Boolean>(binaryLabels);
		this.evaluationValues = new ArrayList<Double>();
		List<Double> evaluationCounts = new ArrayList<Double>();
		this.bestPositionCounts = new HashMap<GridSearch<T, Boolean>.GridPosition, Integer>();
		for (int i = 0; i < this.evaluations.size(); i++) {
			this.evaluationValues.add(0.0);
			evaluationCounts.add(0.0);
		}
		
		List<SupervisedModel<T, Boolean>> trainedModels = new ArrayList<SupervisedModel<T, Boolean>>();
		List<LabelIndicator<L>> trainedLabelIndicators = new ArrayList<LabelIndicator<L>>();
		for (int i = 0; i < this.binaryValidations.size(); i++) {
			ValidationGST<T, Boolean> validation = this.binaryValidations.get(i);
			if (this.binaryValidations.get(i).trainData.getDataSizeForLabel(true) == 0
					|| this.binaryValidations.get(i).devData.getDataSizeForLabel(true) == 0
					|| (this.binaryValidations.get(i).testData != null && this.binaryValidations.get(i).testData.getDataSizeForLabel(true) == 0)) {
				output.resultsWriteln("Ignored " + this.trainData.getDatumTools().getLabelIndicators().get(i).toString() + " (lacking positive examples)");
				continue;
			}
			
			
			if (validation == null) {
				output.debugWriteln("ERROR: Validation thread failed.");
				return null;
			}
			
			ConfusionMatrix<T, Boolean> confusions = validation.getConfusionMatrix();
			
			for (int j = 0; j < this.evaluations.size(); j++) {
				this.evaluationValues.set(j, this.evaluationValues.get(j) + validation.getEvaluationValues().get(j));
				evaluationCounts.set(j, evaluationCounts.get(j) + 1); // FIXME: This is no longer necessary
			}
			
			this.aggregateConfusionMatrix.add(confusions);
			
			if (!this.bestPositionCounts.containsKey(validation.bestGridPosition))
				this.bestPositionCounts.put(validation.bestGridPosition, 0);
			this.bestPositionCounts.put(validation.bestGridPosition, this.bestPositionCounts.get(validation.bestGridPosition) + 1);
		
			trainedModels.add(validation.getModel());
			trainedLabelIndicators.add(this.trainData.getDatumTools().getLabelIndicators().get(i));
		}
		
		for (int j = 0; j < this.evaluations.size(); j++) {
			this.evaluationValues.set(j, this.evaluationValues.get(j) / evaluationCounts.get(j));
		}
		
		this.learnedCompositeModel = new SupervisedModelCompositeBinary<T, D, L>(
				trainedModels, 
				trainedLabelIndicators, 
				this.binaryValidations.get(0).datumTools, 
				this.inverseLabelIndicator);
		
		Map<D, L> classifiedData = this.learnedCompositeModel.classify(this.testData);
		for (int j = 0; j < this.compositeEvaluations.size(); j++)
			this.compositeEvaluationValues.add(this.compositeEvaluations.get(j).evaluate(this.learnedCompositeModel, this.testData, classifiedData));
		
		return this.evaluationValues;
	}
	
	@Override
	public boolean outputModel() {
		return true;
	}
	
	@Override
	public boolean outputData() { 
		return true;
	}
	
	@Override
	public boolean outputResults() {
		OutputWriter output = this.datumTools.getDataTools().getOutputWriter();
		DecimalFormat cleanDouble = new DecimalFormat("0.00000");
		
		output.resultsWrite("\nMeasures:\t");
		for (SupervisedModelEvaluation<D, L> evaluation : this.evaluations)
			output.resultsWrite(evaluation.toString(false) + "\t");
		
		for (ValidationGST<T, Boolean> validation : this.binaryValidations) {
			if (validation.getEvaluationValues() == null)
				continue;
			
			output.resultsWrite("\n" + validation.name + ":\t");
			
			for (Double value : validation.getEvaluationValues())
				output.resultsWrite(cleanDouble.format(value) + "\t");
		}
		
		output.resultsWrite("\nAverages:\t");
		for (Double value : this.evaluationValues) 
			output.resultsWrite(cleanDouble.format(value) + "\t");
		
		output.resultsWriteln("\n\nConfusion matrix:\n" + this.aggregateConfusionMatrix.toString());
		
		output.resultsWriteln("\nBest grid search position counts:");
		for (Entry<GridSearch<T, Boolean>.GridPosition, Integer> entry : this.bestPositionCounts.entrySet())
			output.resultsWriteln(entry.getKey().toString() + "\t" + entry.getValue());	
		
		output.resultsWriteln("\nComposite evaluations:");
		for (int i = 0; i < this.compositeEvaluations.size(); i++) {
			output.resultsWriteln(this.compositeEvaluations.get(i).toString(false) + "\t" + cleanDouble.format(this.compositeEvaluationValues.get(i)));
		}
		
		output.resultsWriteln("\nTime:\n" + this.datumTools.getDataTools().getTimer().toString());
	
		return true;
	}
	
	@Override
	public String getParameterValue(String parameter) {
		return super.getParameterValue(parameter);
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue,
			Tools<D, L> datumTools) {
		return super.setParameterValue(parameter, parameterValue, datumTools);
	}
	
	@Override
	public boolean deserializeNext(BufferedReader reader, String nextName) throws IOException {
		if (nextName.equals("compositeEvaluation")) {
			String evaluationName = SerializationUtil.deserializeGenericName(reader);
			SupervisedModelEvaluation<D, L> evaluation = this.datumTools.makeEvaluationInstance(evaluationName);
			if (!evaluation.deserialize(reader, false, this.datumTools))
				return false;
			this.compositeEvaluations.add(evaluation);
		} else {
			return super.deserializeNext(reader, nextName);
		}
		
		return true;
	}
}
