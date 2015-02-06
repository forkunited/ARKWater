package ark.model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;

import org.apache.commons.io.input.ReaderInputStream;
import org.apache.commons.io.output.WriterOutputStream;
import org.platanios.learn.classification.LogisticRegressionAdaGrad;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorNorm;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.OutputWriter;
import ark.util.Pair;

public class SupervisedModelAreg<D extends Datum<L>, L> extends SupervisedModel<D, L> {
	private double l1;
	private double l2;
	private double convergenceEpsilon = -1.0;
	private int maxEvaluationConstantIterations = 100000;
	private double maxTrainingExamples = 10000;
	private int batchSize = 100;
	private int evaluationIterations = 500;
	private boolean weightedLabels = false;
	private double classificationThreshold = 0.5;
	private String[] hyperParameterNames = { "l1", "l2", "convergenceEpsilon", "maxEvaluationConstantIterations", "maxTrainingExamples", "batchSize", "evaluationIterations", "weightedLabels", "classificationThreshold" };
	
	private LogisticRegressionAdaGrad classifier;
	private Vector classifierWeights;
	private FeaturizedDataSet<D, L> trainingData;
	
	@Override
	public boolean train(FeaturizedDataSet<D, L> data, FeaturizedDataSet<D, L> testData, List<SupervisedModelEvaluation<D, L>> evaluations) {
		OutputWriter output = data.getDatumTools().getDataTools().getOutputWriter();
		
		if (this.validLabels.size() > 2 || !this.validLabels.contains(true)) {
			output.debugWriteln("ERROR: Areg only supports binary classification.");
			return false;
		}
		
		DataSet<PredictedDataInstance<Vector, Double>> plataniosData = data.makePlataniosDataSet(this.weightedLabels, 1.0/5.0, true);
		
		List<Double> initEvaluationValues = new ArrayList<Double>();
		for (int i = 0; i < evaluations.size(); i++) {
			initEvaluationValues.add(0.0);
		}
		
		//int iterationsPerDataSet = plataniosData.size()/this.batchSize;
		int maximumIterations =  (int)(this.maxTrainingExamples/this.batchSize);// (int)Math.floor(this.maxDataSetRuns*iterationsPerDataSet);		
		
		NumberFormat format = new DecimalFormat("#0.000000");
		
		output.debugWriteln("Areg training platanios model...");
		
		SupervisedModel<D, L> thisModel = this;
		this.classifier =
				new LogisticRegressionAdaGrad.Builder(plataniosData.get(0).features().size())
					.sparse(true)
					.useBiasTerm(true)
					.useL1Regularization(this.l1 > 0)
					.useL2Regularization(this.l2 > 0)
					.l1RegularizationWeight(this.l1)
					.l2RegularizationWeight(this.l2)
					.batchSize(this.batchSize)
					.maximumNumberOfIterations(maximumIterations)
					.pointChangeTolerance(this.convergenceEpsilon)
					.additionalCustomConvergenceCriterion(new Function<Vector, Boolean>() {
						int iterations = 0;
						int evaluationConstantIterations = 0;
						Map<D, L> prevPredictions = null;
						List<Double> prevEvaluationValues = initEvaluationValues;
						SupervisedModel<D, L> model = thisModel;
						Vector prevWeights = null;
						
						@Override
						public Boolean apply(Vector weights) {
							this.iterations++;
							
							if (this.iterations % evaluationIterations != 0) {
								this.prevWeights = weights;
								classifierWeights = weights;
								return false;
							}
								
							LogisticRegressionAdaGrad tempClassifier = classifier;
							classifier = new LogisticRegressionAdaGrad.Builder(plataniosData.get(0).features().size(), weights)
								.sparse(true)
								.useBiasTerm(true)
								.build();
							
							Map<D, L> predictions = classify(testData);
							int labelDifferences = countLabelDifferences(prevPredictions, predictions);
							List<Double> evaluationValues = new ArrayList<Double>();
							for (SupervisedModelEvaluation<D, L> evaluation : evaluations) {
								evaluationValues.add(evaluation.evaluate(model, testData, predictions));
							}
							
							classifier = tempClassifier;
							
							double pointChange = weights.sub(this.prevWeights).norm(VectorNorm.L2_FAST);
							
							String amountDoneStr = format.format(this.iterations/(double)maximumIterations);
							String pointChangeStr = format.format(pointChange);
							String statusStr = data.getName() + " (l1=" + l1 + ", l2=" + l2 + ") #" + iterations + 
									" [" + amountDoneStr + "] -- point-change: " + pointChangeStr + " predict-diff: " + labelDifferences + "/" + predictions.size() + " ";
							for (int i = 0; i < evaluations.size(); i++) {
								String evaluationName = evaluations.get(i).getGenericName();
								String evaluationDiffStr = format.format(evaluationValues.get(i) - this.prevEvaluationValues.get(i));
								String evaluationValueStr= format.format(evaluationValues.get(i));
								statusStr += evaluationName + " diff: " + evaluationDiffStr + " " + evaluationName + ": " + evaluationValueStr + " ";
							}
							output.debugWriteln(statusStr);
							
							double evaluationDiff = evaluationValues.get(0) - this.prevEvaluationValues.get(0);
							if (Double.compare(evaluationDiff, 0.0) == 0) {
								this.evaluationConstantIterations += evaluationIterations;
							} else {
								this.evaluationConstantIterations = 0;
							}
								
							this.prevPredictions = predictions;
							this.prevEvaluationValues = evaluationValues;
							this.prevWeights = weights;
							classifierWeights = weights;
							
							if (maxEvaluationConstantIterations < this.evaluationConstantIterations)
								return true;
							
							return false;
						}
						
						private int countLabelDifferences(Map<D, L> labels1, Map<D, L> labels2) {
							if (labels1 == null && labels2 != null)
								return labels2.size();
							if (labels1 != null && labels2 == null)
								return labels1.size();
							if (labels1 == null && labels2 == null)
								return 0;
							
							int count = 0;
							for (Entry<D, L> entry: labels1.entrySet()) {
								if (!labels2.containsKey(entry.getKey()) || !entry.getValue().equals(labels2.get(entry.getKey())))
									count++;
							}
							return count;
						}
						
					})
					.loggingLevel(0)
					.random(data.getDatumTools().getDataTools().makeLocalRandom())
					.build();

		output.debugWriteln(data.getName() + " training for at most " + maximumIterations + 
				" iterations (maximum " + this.maxTrainingExamples + " examples over size " + this.batchSize + " batches from " + data.size() + " examples)");
		
		if (!this.classifier.train(plataniosData)) {
			output.debugWriteln("ERROR: Areg failed to train platanios model.");
			return false;
		}
		
		this.trainingData = data;
		
		output.debugWriteln("Areg finished training platanios model."); 
		
		return true;
	}

	@SuppressWarnings("unchecked")
	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		OutputWriter output = data.getDatumTools().getDataTools().getOutputWriter();

		if (this.validLabels.size() > 2 || !this.validLabels.contains(true)) {
			output.debugWriteln("ERROR: Areg only supports binary classification.");
			return null;
		}
		
		DataSet<PredictedDataInstance<Vector, Double>> plataniosData = data.makePlataniosDataSet(this.weightedLabels, 0.0, false);

		plataniosData = this.classifier.predict(plataniosData);
		if (plataniosData == null) {
			output.debugWriteln("ERROR: Areg failed to compute data posteriors.");
			return null;
		}
		
		Map<D, Map<L, Double>> posteriors = new HashMap<D, Map<L, Double>>();
		for (PredictedDataInstance<Vector, Double> prediction : plataniosData) {
			int datumId = Integer.parseInt(prediction.name());
			D datum = data.getDatumById(datumId);
			
			Map<L, Double> posterior = new HashMap<L, Double>();
			double p = (prediction.label() == 1) ? prediction.probability() : 1.0 - prediction.probability();
			
			// Offset bias term according to classification threshold so that
			// output posterior p >= 0.5 iff model_p >= classification threshold
			// i.e. log (p/(1-p)) = log (model_p/(1-model_p)) - log (threshold/(1-threshold))
			p = p*(1.0-this.classificationThreshold)/(this.classificationThreshold*(1.0-p)+p*(1.0-this.classificationThreshold));
			
			posterior.put((L)(new Boolean(true)), p);
			posterior.put((L)(new Boolean(false)), 1.0-p);
			
			posteriors.put(datum, posterior);
		}

		return posteriors;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public Map<D, L> classify(FeaturizedDataSet<D, L> data) {
		OutputWriter output = data.getDatumTools().getDataTools().getOutputWriter();

		if (this.validLabels.size() > 2 || !this.validLabels.contains(true)) {
			output.debugWriteln("ERROR: Areg only supports binary classification.");
			return null;
		}
		
		DataSet<PredictedDataInstance<Vector, Double>> plataniosData = data.makePlataniosDataSet(this.weightedLabels, 0.0, false);

		plataniosData = this.classifier.predict(plataniosData);
		if (plataniosData == null) {
			output.debugWriteln("ERROR: Areg failed to compute data classifications.");
			return null;
		}
		
		Map<D, Boolean> predictions = new HashMap<D, Boolean>();
		for (PredictedDataInstance<Vector, Double> prediction : plataniosData) {
			int datumId = Integer.parseInt(prediction.name());
			D datum = data.getDatumById(datumId);
		
			if (this.fixedDatumLabels.containsKey(datum)) {
				predictions.put(datum, (Boolean)this.fixedDatumLabels.get(datum));
				continue;
			}
			
			double p = (prediction.label() == 1) ? prediction.probability() : 1.0 - prediction.probability();
			if (p >= this.classificationThreshold)
				predictions.put(datum, true);
			else 
				predictions.put(datum, false);
		}
		
		return (Map<D, L>)predictions;
	}
	
	@Override
	protected boolean deserializeParameters(BufferedReader reader, Tools<D, L> datumTools) {
		try {
			String line = null;
			while((line = reader.readLine()) != null)
				if (line.equals("# END WEIGHTS #"))
					break;
				
			this.classifier = LogisticRegressionAdaGrad.read(new ReaderInputStream(reader), true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return this.classifier != null;
	}

	@Override
	protected boolean serializeParameters(Writer writer) throws IOException {
		writer.write("# WEIGHTS #\n");
		
		double[] weightArray = this.classifierWeights.getDenseArray(); 
		writer.write("featureVocabularySize=" + weightArray.length + "\n");
		
		List<Pair<Integer, Double>> sortedWeights = new ArrayList<Pair<Integer, Double>>();
		List<Integer> nonZeroWeightIndices = new ArrayList<Integer>();
		for (int i = 0; i < weightArray.length - 1; i++) {
			if (Double.compare(weightArray[i], 0) != 0) {
				sortedWeights.add(new Pair<Integer, Double>(i, weightArray[i]));
				nonZeroWeightIndices.add(i);
			}
		}
		
		writer.write("nonZeroWeights=" + nonZeroWeightIndices.size() + "\n");
		
		Map<Integer, String> featureNames = this.trainingData.getFeatureVocabularyNamesForIndices(nonZeroWeightIndices);
		
		Collections.sort(sortedWeights, new Comparator<Pair<Integer, Double>>() {
			@Override
			public int compare(Pair<Integer, Double> w1,
					Pair<Integer, Double> w2) {
				if (Math.abs(w1.getSecond()) > Math.abs(w2.getSecond()))
					return -1;
				else if (Math.abs(w1.getSecond()) < Math.abs(w2.getSecond()))
					return 1;
				else
					return 0;
			} });
		
		writer.write("bias=" + weightArray[weightArray.length - 1] + "\n");
		for (Pair<Integer, Double> weight : sortedWeights) {
			double w = weight.getSecond();
			int index = weight.getFirst();
			
			String featureStr = featureNames.get(index) + "(w=" + w + ", index=" + index + ")";
			writer.write(featureStr + "\n");
		}
		
		writer.write("# END WEIGHTS #\n");
		
		this.classifier.write(new WriterOutputStream(writer), true);
		
		return true;
	}

	@Override
	protected boolean serializeExtraInfo(Writer writer) {
		return true;
	}

	@Override
	public String getGenericName() {
		return "Areg";
	}
	
	@Override
	public String getParameterValue(String parameter) {
		if (parameter.equals("l1"))
			return String.valueOf(this.l1);
		else if (parameter.equals("l2"))
			return String.valueOf(this.l2);
		else if (parameter.equals("convergenceEpsilon"))
			return String.valueOf(this.convergenceEpsilon);
		else if (parameter.equals("maxTrainingExamples"))
			return String.valueOf(this.maxTrainingExamples);
		else if (parameter.equals("batchSize"))
			return String.valueOf(this.batchSize);
		else if (parameter.equals("evaluationIterations"))
			return String.valueOf(this.evaluationIterations);
		else if (parameter.equals("maxEvaluationConstantIterations"))
			return String.valueOf(this.maxEvaluationConstantIterations);
		else if (parameter.equals("weightedLabels"))
			return String.valueOf(this.weightedLabels);
		else if (parameter.equals("classificationThreshold"))
			return String.valueOf(this.classificationThreshold);
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("l1"))
			this.l1 = Double.valueOf(parameterValue);
		else if (parameter.equals("l2"))
			this.l2 = Double.valueOf(parameterValue);
		else if (parameter.equals("convergenceEpsilon"))
			this.convergenceEpsilon = Double.valueOf(parameterValue);
		else if (parameter.equals("maxTrainingExamples"))
			this.maxTrainingExamples = Double.valueOf(parameterValue);
		else if (parameter.equals("batchSize"))
			this.batchSize = Integer.valueOf(parameterValue);
		else if (parameter.equals("evaluationIterations"))
			this.evaluationIterations = Integer.valueOf(parameterValue);
		else if (parameter.equals("maxEvaluationConstantIterations"))
			this.maxEvaluationConstantIterations = Integer.valueOf(parameterValue);
		else if (parameter.equals("weightedLabels"))
			this.weightedLabels = Boolean.valueOf(parameterValue);
		else if (parameter.equals("classificationThreshold"))
			this.classificationThreshold = Double.valueOf(parameterValue);
		else
			return false;
		return true;
	}
	
	@Override
	public String[] getParameterNames() {
		return this.hyperParameterNames;
	}

	@Override
	protected SupervisedModel<D, L> makeInstance() {
		return new SupervisedModelAreg<D, L>();
	}

	@Override
	protected boolean deserializeExtraInfo(String name, BufferedReader reader, Tools<D, L> datumTools) {
		return true;
	}
}
