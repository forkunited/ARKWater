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

import org.apache.commons.io.input.ReaderInputStream;
import org.apache.commons.io.output.WriterOutputStream;
import org.platanios.learn.classification.LogisticRegressionAdaGrad;
import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

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
	private double maxDataSetRuns = 1.0;
	private int batchSize = 100;
	private int evaluationIterations;
	private String[] hyperParameterNames = { "l1", "l2", "convergenceEpsilon", "maxEvaluationConstantIterations", "maxDataSetRuns", "batchSize", "evaluationIterations" };
	private LogisticRegressionAdaGrad classifier;
	private FeaturizedDataSet<D, L> trainingData;
	
	@Override
	public boolean train(FeaturizedDataSet<D, L> data, FeaturizedDataSet<D, L> testData, List<SupervisedModelEvaluation<D, L>> evaluations) {
		OutputWriter output = data.getDatumTools().getDataTools().getOutputWriter();
		
		if (this.validLabels.size() > 2 || !this.validLabels.contains(true)) {
			output.debugWriteln("ERROR: Areg only supports binary classification.");
			return false;
		}
		
		DataSet<PredictedDataInstance<Vector, Integer>> plataniosData = data.makePlataniosDataSet();
		
		output.debugWriteln("Areg training platanios model...");
		
		this.classifier =
				new LogisticRegressionAdaGrad.Builder(plataniosData.get(0).features().size())
					.sparse(true)
					.useBiasTerm(true)
					.useL1Regularization(this.l1 > 0)
					.useL2Regularization(this.l2 > 0)
					.l1RegularizationWeight(this.l1)
					.l2RegularizationWeight(this.l2)
					.batchSize(this.batchSize)
					.maximumNumberOfIterations(this.evaluationIterations)
					.pointChangeTolerance(this.convergenceEpsilon)
					.loggingLevel(0)
					.random(data.getDatumTools().getDataTools().makeLocalRandom())
					.build();
		
		Map<D, L> prevPredictions = classify(testData);
		List<Double> prevEvaluationValues = new ArrayList<Double>();
		for (SupervisedModelEvaluation<D, L> evaluation : evaluations) {
			prevEvaluationValues.add(evaluation.evaluate(this, testData, prevPredictions));
		}
		
		int iterationsPerDataSet = data.size()/this.batchSize;
		int maximumIterations = (int)Math.floor(this.maxDataSetRuns*iterationsPerDataSet);
		
		output.debugWriteln(data.getName() + " training for at most " + maximumIterations + 
				" iterations (maximum " + this.maxDataSetRuns + " runs over size " + this.batchSize + " batches from " + data.size() + " examples)");
		
		int iterations = 0;
		double pointChange = Double.MAX_VALUE;
		int evaluationConstantIterations = 0;
		
		NumberFormat format = new DecimalFormat("#0.000"); 
		
		while (iterations < maximumIterations && pointChange > this.convergenceEpsilon && evaluationConstantIterations <= this.maxEvaluationConstantIterations) { 
			if (!this.classifier.train(plataniosData)) {
				output.debugWriteln("ERROR: Areg failed to train platanios model.");
				return false;
			}
			
			Map<D, L> predictions = classify(testData);
			int labelDifferences = countLabelDifferences(prevPredictions, predictions);
			List<Double> evaluationValues = new ArrayList<Double>();
			for (SupervisedModelEvaluation<D, L> evaluation : evaluations) {
				evaluationValues.add(evaluation.evaluate(this, testData, predictions));
			}
			pointChange = this.classifier.solver().getPointChange();
			iterations += this.evaluationIterations;
			
			String amountDoneStr = format.format(iterations/(double)maximumIterations);
			String pointChangeStr = format.format(pointChange);
			String statusStr = data.getName() + " (l1=" + this.l1 + ", l2=" + this.l2 + ") #" + iterations + 
					" [" + amountDoneStr + "] -- point-change: " + pointChangeStr + " predict-diff: " + labelDifferences + "/" + predictions.size() + " ";
			for (int i = 0; i < evaluations.size(); i++) {
				String evaluationName = evaluations.get(i).getGenericName();
				String evaluationDiffStr = format.format(evaluationValues.get(i) - prevEvaluationValues.get(i));
				String evaluationValueStr= format.format(evaluationValues.get(i));
				statusStr += evaluationName + " diff: " + evaluationDiffStr + " " + evaluationName + ": " + evaluationValueStr + " ";
			}
			output.debugWriteln(statusStr);
			
			double evaluationDiff = evaluationValues.get(0) - prevEvaluationValues.get(0);
			if (Double.compare(evaluationDiff, 0.0) == 0) {
				evaluationConstantIterations += this.evaluationIterations;
			} else {
				evaluationConstantIterations = 0;
			}
			
			prevPredictions = predictions;
			prevEvaluationValues = evaluationValues;
		}
		
		this.trainingData = data;
		
		output.debugWriteln("Areg finished training platanios model."); 
		
		return true;
	}
	
	private int countLabelDifferences(Map<D, L> labels1, Map<D, L> labels2) {
		int count = 0;
		for (Entry<D, L> entry: labels1.entrySet()) {
			if (!labels2.containsKey(entry.getKey()) || !entry.getValue().equals(labels2.get(entry.getKey())))
				count++;
		}
		return count;
	}

	@SuppressWarnings("unchecked")
	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		OutputWriter output = data.getDatumTools().getDataTools().getOutputWriter();

		if (this.validLabels.size() > 2 || !this.validLabels.contains(true)) {
			output.debugWriteln("ERROR: Areg only supports binary classification.");
			return null;
		}
		
		DataSet<PredictedDataInstance<Vector, Integer>> plataniosData = data.makePlataniosDataSet();

		plataniosData = this.classifier.predict(plataniosData);
		if (plataniosData == null) {
			output.debugWriteln("ERROR: Areg failed to compute data posteriors.");
			return null;
		}
		
		Map<D, Map<L, Double>> posteriors = new HashMap<D, Map<L, Double>>();
		for (PredictedDataInstance<Vector, Integer> prediction : plataniosData) {
			int datumId = Integer.parseInt(prediction.name());
			D datum = data.getDatumById(datumId);
			
			Map<L, Double> posterior = new HashMap<L, Double>();
			if (prediction.label() == 1) {
				posterior.put((L)(new Boolean(true)), prediction.probability());
				posterior.put((L)(new Boolean(false)), 1.0 - prediction.probability());
			} else {
				posterior.put((L)(new Boolean(false)), prediction.probability());
				posterior.put((L)(new Boolean(true)), 1.0 - prediction.probability());
			}
			
			posteriors.put(datum, posterior);
		}

		return posteriors;
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
		
		double[] weightArray = this.classifier.weights().getDenseArray(); 
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
		else if (parameter.equals("maxDataSetRuns"))
			return String.valueOf(this.maxDataSetRuns);
		else if (parameter.equals("batchSize"))
			return String.valueOf(this.batchSize);
		else if (parameter.equals("evaluationIterations"))
			return String.valueOf(this.evaluationIterations);
		else if (parameter.equals("maxEvaluationConstantIterations"))
			return String.valueOf(this.maxEvaluationConstantIterations);
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
		else if (parameter.equals("maxDataSetRuns"))
			this.maxDataSetRuns = Double.valueOf(parameterValue);
		else if (parameter.equals("batchSize"))
			this.batchSize = Integer.valueOf(parameterValue);
		else if (parameter.equals("evaluationIterations"))
			this.evaluationIterations = Integer.valueOf(parameterValue);
		else if (parameter.equals("maxEvaluationConstantIterations"))
			this.maxEvaluationConstantIterations = Integer.valueOf(parameterValue);
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
