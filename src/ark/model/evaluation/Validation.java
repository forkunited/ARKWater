package ark.model.evaluation;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.TokenSpanExtractor;
import ark.data.feature.Feature;
import ark.model.SupervisedModel;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.FileUtil;
import ark.util.OutputWriter;
import ark.util.Parameterizable;
import ark.util.SerializationUtil;

public abstract class Validation<D extends Datum<L>, L> implements Parameterizable<D, L> {
	protected String name;
	protected Datum.Tools<D, L> datumTools;
	protected int maxThreads;
	protected SupervisedModel<D, L> model;
	protected TokenSpanExtractor<D, L> errorExampleExtractor;
	protected List<SupervisedModelEvaluation<D, L>> evaluations;
	
	protected ConfusionMatrix<D, L> confusionMatrix;
	protected List<Double> evaluationValues;
	
	public Validation(String name, Datum.Tools<D, L> datumTools) {
		this.name = name;
		this.datumTools = datumTools;
		this.evaluations = new ArrayList<SupervisedModelEvaluation<D, L>>();
	}
	
	public Validation(String name, Datum.Tools<D, L> datumTools, int maxThreads, SupervisedModel<D, L> model, List<SupervisedModelEvaluation<D, L>> evaluations, TokenSpanExtractor<D, L> errorExampleExtractor) {
		this.name = name;
		this.datumTools = datumTools;
		this.maxThreads = maxThreads;
		this.model = model;
		this.evaluations = evaluations;
		this.errorExampleExtractor = errorExampleExtractor;
	}
	
	public boolean runAndOutput(String inputPath) {
		return deserialize(inputPath) 
				&& run() != null
				&& outputAll();
	}
	
	public boolean runAndOutput() {
		return run() != null && outputAll();
	}
	
	public boolean outputAll() {
		return outputModel() && outputData() && outputResults();
	}
	
	public boolean outputResults() {
		OutputWriter output = this.datumTools.getDataTools().getOutputWriter();
		
		output.resultsWriteln("\nEvaluation results:");
		
		for (int i = 0; i < this.evaluations.size(); i++)
			output.resultsWriteln(this.evaluations.get(i).toString() + ": " + this.evaluationValues.get(i));
		
		output.resultsWriteln("\nConfusion matrix:\n" + this.confusionMatrix.toString());
		
		output.resultsWriteln("\nTime:\n" + this.datumTools.getDataTools().getTimer().toString());
		
		return true;
	}
	
	public boolean outputModel() {
		this.datumTools.getDataTools().getOutputWriter().modelWriteln(this.model.toString(true));
		return true;
	}
	
	public boolean outputData() {
		this.datumTools.getDataTools().getOutputWriter().dataWriteln(this.confusionMatrix.getActualToPredictedDescription(this.errorExampleExtractor));
		return true;
	}
	
	public List<SupervisedModelEvaluation<D, L>> getEvaluations() {
		return this.evaluations;
	}
	
	public SupervisedModel<D, L> getModel() {
		return this.model;
	}
	
	public ConfusionMatrix<D, L> getConfusionMatrix() {
		return this.confusionMatrix;
	}
	
	public List<Double> getEvaluationValues() {
		return this.evaluationValues;
	}
	
	public String getErrorExamples() {
		return this.confusionMatrix.getActualToPredictedDescription(this.errorExampleExtractor);
	}
	
	public boolean deserializeNext(BufferedReader reader, String nextName) throws IOException {
		return setParameterValue(nextName, SerializationUtil.deserializeAssignmentRight(reader), this.datumTools);
	}
	
	public boolean deserialize(String path) {
		return deserialize(FileUtil.getFileReader(path));
	}
	
	public boolean deserialize(BufferedReader reader) {
		String assignmentLeft = null; // The name on the left hand side of the equals sign
		
		try {
			while ((assignmentLeft = SerializationUtil.deserializeAssignmentLeft(reader)) != null) {
				if (assignmentLeft.equals("randomSeed"))
					this.datumTools.getDataTools().setRandomSeed(Long.valueOf(SerializationUtil.deserializeAssignmentRight(reader)));
				else if (assignmentLeft.equals("maxThreads")) {
					if (!setMaxThreads(Integer.valueOf(SerializationUtil.deserializeAssignmentRight(reader))))
						return false;
				} else if (assignmentLeft.startsWith("model")) {
					String[] nameParts = assignmentLeft.split("_");
					String referenceName = null;
					if (nameParts.length > 1)
						referenceName = nameParts[1];
					
					String modelName = SerializationUtil.deserializeGenericName(reader);
					this.model = this.datumTools.makeModelInstance(modelName);
					if (!this.model.deserialize(reader, false, false, this.datumTools, referenceName))
						return false;
				} else if (assignmentLeft.startsWith("feature")) {
					String[] nameParts = assignmentLeft.split("_");
					String referenceName = null;
					boolean ignore = false;
					if (nameParts.length > 1)
						referenceName = nameParts[1];
					if (nameParts.length > 2)
						ignore = true;
					String featureName = SerializationUtil.deserializeGenericName(reader);
					Feature<D, L> feature = this.datumTools.makeFeatureInstance(featureName);
					if (!feature.deserialize(reader, false, false, this.datumTools, referenceName, ignore))
						return false;
					addFeature(feature);
				} else if (assignmentLeft.startsWith("errorExampleExtractor")) {
					this.errorExampleExtractor = this.datumTools.getTokenSpanExtractor(
							SerializationUtil.deserializeAssignmentRight(reader));
				} else if (assignmentLeft.startsWith("evaluation")) {
					String evaluationName = SerializationUtil.deserializeGenericName(reader);
					SupervisedModelEvaluation<D, L> evaluation = this.datumTools.makeEvaluationInstance(evaluationName);
					if (!evaluation.deserialize(reader, false, this.datumTools))
						return false;
					this.evaluations.add(evaluation);
				} else if (!deserializeNext(reader, assignmentLeft))
					return false;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return true;
	}
	
	protected abstract boolean addFeature(Feature<D, L> feature);
	protected abstract boolean setMaxThreads(int maxThreads);
	public abstract List<Double> run();
}
