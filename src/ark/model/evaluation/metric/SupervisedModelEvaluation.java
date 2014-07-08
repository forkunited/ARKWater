package ark.model.evaluation.metric;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelMapping;
import ark.data.feature.FeaturizedDataSet;
import ark.model.SupervisedModel;
import ark.util.Pair;
import ark.util.SerializationUtil;

/**
 * SupervisedModelEvaluation represents evaluation measure
 * for a supervised classification model.  
 * 
 * Implementations of particular evaluation measures derive
 * from SupervisedModelEvaluation, and SupervisedModelEvaluation
 * is primarily responsible for providing the methods necessary
 * for deserializing these evaluation measures from configuration
 * files.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public abstract class SupervisedModelEvaluation<D extends Datum<L>, L> {
	protected LabelMapping<L> labelMapping;
	
	/**
	 * @return a generic name by which to refer to the evaluation
	 */
	public abstract String getGenericName();
	
	/**
	 * 
	 * @param model
	 * @param data
	 * @param predictions
	 * @return the value of the evaluation measure for the model with predictions on the
	 * given data set.  This method should generally start by calling 
	 * 'getMappedActualAndPredictedLabels' to get the labels according to the 'labelMapping'
	 * function, and then compute the measure using those returned labels.
	 * 
	 */
	protected abstract double compute(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data, Map<D, L> predictions);
	
	/**
	 * @return parameters of the evaluation measure that can be set through the experiment 
	 * configuration file
	 */
	protected abstract String[] getParameterNames();
	
	/**
	 * @param parameter
	 * @return the value of the given parameter
	 */
	protected abstract String getParameterValue(String parameter);
	
	/**
	 * 
	 * @param parameter
	 * @param parameterValue
	 * @param datumTools
	 * @return true if the parameter has been set to parameterValue.  Some parameters are set to
	 * objects retrieved through datumTools that are named by parameterValue.
	 */
	protected abstract boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools);
	
	/**
	 * @return a generic instance of the evaluation measure.  This is used when deserializing
	 * the parameters for the measure from a configuration file
	 */
	protected abstract SupervisedModelEvaluation<D, L> makeInstance();
	
	/**
	 * @param predictions
	 * @return a list of pairs of actual predicted labels that are mapped by
	 * the label mapping (labelMapping) from the given predictions map
	 */
	protected List<Pair<L, L>> getMappedActualAndPredictedLabels(Map<D, L> predictions) {
		List<Pair<L, L>> mappedLabels = new ArrayList<Pair<L, L>>(predictions.size());
		for (Entry<D, L> prediction : predictions.entrySet()) {
			L actual = prediction.getKey().getLabel();
			L predicted = prediction.getValue();
			if (this.labelMapping != null) {
				actual = this.labelMapping.map(actual);
				predicted = this.labelMapping.map(predicted);
			}
			
			Pair<L, L> mappedPair = new Pair<L, L>(actual, predicted);
			mappedLabels.add(mappedPair);
		}
		
		return mappedLabels;
	}
	
	/**
	 * 
	 * @param model
	 * @param data
	 * @param predictions
	 * @return the value of the evaluation measure for the model with predictions on the
	 * given data set.  If the model has a label mapping, then it is used when 
	 * computing the evaluation measure.
	 */
	public double evaluate(SupervisedModel<D, L> model, FeaturizedDataSet<D, L> data, Map<D, L> predictions) {
		LabelMapping<L> modelLabelMapping = model.getLabelMapping();
		if (this.labelMapping != null)
			model.setLabelMapping(this.labelMapping);
		
		double evaluation = compute(model, data, predictions);
		model.setLabelMapping(modelLabelMapping);
		
		return evaluation;
	}
	
	public SupervisedModelEvaluation<D, L> clone(Datum.Tools<D, L> datumTools) {
		return clone(datumTools, null);
	}
	
	public SupervisedModelEvaluation<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		SupervisedModelEvaluation<D, L> clone = makeInstance();
		String[] parameterNames = getParameterNames();
		for (int i = 0; i < parameterNames.length; i++) {
			String parameterValue = getParameterValue(parameterNames[i]);
			if (environment != null && parameterValue != null) {
				for (Entry<String, String> entry : environment.entrySet())
					parameterValue = parameterValue.replace("${" + entry.getKey() + "}", entry.getValue());
			}
			clone.setParameterValue(parameterNames[i], parameterValue, datumTools);
		}
		return clone;
	}
	
	public boolean deserialize(BufferedReader reader, boolean readGenericName, Datum.Tools<D, L> datumTools) throws IOException {
		if (readGenericName && SerializationUtil.deserializeGenericName(reader) == null)
			return false;
		
		Map<String, String> parameters = SerializationUtil.deserializeArguments(reader);
		if (parameters != null)
			for (Entry<String, String> entry : parameters.entrySet()) {
				if (entry.getKey().equals("labelMapping"))
					this.labelMapping = datumTools.getLabelMapping(entry.getValue());
				else
					setParameterValue(entry.getKey(), entry.getValue(), datumTools);
			}

		return true;
	}
	
	public boolean serialize(Writer writer) throws IOException {
		writer.write(toString(false));
		return true;
	}
	
	public String toString(boolean withVocabulary) {
		if (withVocabulary) {
			StringWriter stringWriter = new StringWriter();
			try {
				if (serialize(stringWriter))
					return stringWriter.toString();
				else
					return null;
			} catch (IOException e) {
				return null;
			}
		} else {
			String genericName = getGenericName();
			Map<String, String> parameters = new HashMap<String, String>();
			String[] parameterNames = getParameterNames();
			for (int i = 0; i < parameterNames.length; i++)
				parameters.put(parameterNames[i], getParameterValue(parameterNames[i]));
			
			if (this.labelMapping != null)
				parameters.put("labelMapping", this.labelMapping.toString());
			
			StringWriter parametersWriter = new StringWriter();
			
			try {
				SerializationUtil.serializeArguments(parameters, parametersWriter);
			} catch (IOException e) {
				return null;
			}
			
			String parametersStr = parametersWriter.toString();
			return genericName + "(" + parametersStr + ")";
		}
	}
	
	public String toString() {
		return toString(false);
	}
	
	
	public boolean fromString(String str, Datum.Tools<D, L> datumTools) {
		try {
			return deserialize(new BufferedReader(new StringReader(str)), true, datumTools);
		} catch (IOException e) {
			
		}
		return true;
	}
	
}
