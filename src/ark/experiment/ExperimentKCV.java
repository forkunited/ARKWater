package ark.experiment;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;
import ark.data.feature.Feature;
import ark.model.SupervisedModel;
import ark.model.evaluation.KFoldCrossValidation;
import ark.model.evaluation.metric.ClassificationEvaluation;
import ark.util.SerializationUtil;

public class ExperimentKCV<D extends Datum<L>, L> extends Experiment<D, L> {
	protected SupervisedModel<D, L> model;
	protected List<Feature<D, L>> features;
	protected int crossValidationFolds;
	protected Datum.Tools.TokenSpanExtractor<D, L> errorExampleExtractor;
	protected Map<String, List<String>> gridSearchParameterValues;
	protected List<ClassificationEvaluation<D, L>> evaluations;
	protected DataSet<D, L> data;
	
	public ExperimentKCV(String name, String inputPath, DataSet<D, L> data) {
		super(name, inputPath, data.getDatumTools());
		
		this.features = new ArrayList<Feature<D, L>>();
		this.gridSearchParameterValues = new HashMap<String, List<String>>();
		this.evaluations = new ArrayList<ClassificationEvaluation<D, L>>();
		this.data = data;
	}
	
	@Override
	protected boolean execute() {
		KFoldCrossValidation<D, L> validation = new KFoldCrossValidation<D, L>(
			this.name,
			this.model,
			this.features,
			this.evaluations,
			this.data,
			this.crossValidationFolds, 
			this.random
		);
		
		validation.setPossibleHyperParameterValues(this.gridSearchParameterValues);
		
		if (validation.run(this.maxThreads, this.errorExampleExtractor).get(0) < 0)
			return false;

		return true;
	}
	@Override
	protected boolean deserializeNext(BufferedReader reader, String nextName) throws IOException {
		if (nextName.equals("crossValidationFolds")) {
			this.crossValidationFolds = Integer.valueOf(SerializationUtil.deserializeAssignmentRight(reader));
		
		} else if (nextName.startsWith("model")) {
			String modelName = SerializationUtil.deserializeGenericName(reader);
			this.model = this.datumTools.makeModelInstance(modelName);
			if (!this.model.deserialize(reader, false, false, this.datumTools))
				return false;
		} else if (nextName.startsWith("feature")) {
			String featureName = SerializationUtil.deserializeGenericName(reader);
			Feature<D, L> feature = this.datumTools.makeFeatureInstance(featureName);
			if (!feature.deserialize(reader, false, false, this.datumTools))
				return false;
			this.features.add(feature);
		} else if (nextName.startsWith("errorExampleExtractor")) {
			this.errorExampleExtractor = this.datumTools.getTokenSpanExtractor(
					SerializationUtil.deserializeAssignmentRight(reader));
			
		} else if (nextName.startsWith("gridSearchParameterValues")) {
			String parameterName = SerializationUtil.deserializeGenericName(reader);
			List<String> parameterValues = SerializationUtil.deserializeList(reader);
			this.gridSearchParameterValues.put(parameterName, parameterValues);
		
		} else if (nextName.startsWith("evaluation")) {
			String evaluationName = SerializationUtil.deserializeGenericName(reader);
			ClassificationEvaluation<D, L> evaluation = this.datumTools.makeEvaluationInstance(evaluationName);
			if (!evaluation.deserialize(reader, false, this.datumTools))
				return false;
			this.evaluations.add(evaluation);
		}
		
		return true;
	}
}