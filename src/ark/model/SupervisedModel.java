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

package ark.model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelMapping;
import ark.data.feature.FeaturizedDataSet;
import ark.model.evaluation.metric.SupervisedModelEvaluation;
import ark.util.Parameterizable;
import ark.util.SerializationUtil;

/**
 * SupervisedModel represents a supervised classification model
 * that can be trained and evaluated using 
 * ark.data.feature.FeaturizedDataSets.
 * 
 * Implementations of particular supervised models derive from the 
 * SupervisedModel class,
 * and the SupervisedModel class is primarily responsible for 
 * providing the
 * methods necessary for deserializing models from configuration files.
 * Models are defined in the configuration files by strings of
 * the form:
 * 
 * model_[modelReferenceName]=[genericModelName (e.g. SVM)]([hyper-parameter 1]=[hyper-parameter value 1],...)
 * {
 *    [extra info 1]=[extra info value 1]
 *    [...]
 * }
 * 
 * Where strings in square brackets are replaced by model specific info.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public abstract class SupervisedModel<D extends Datum<L>, L> implements Parameterizable<D, L> {
	// Name by which the this particular model instance is referenced
	// This is currently just used by SupervisedModelPartition to
	// assign features to particular models in the experiment configuration
	// file
	private String referenceName;
	
	protected Set<L> validLabels; // Labels that this model can assign
	protected LabelMapping<L> labelMapping; // Mapping from actual labels into valid labels
	
	// Labels that the model should assign regardless of its training
	protected Map<D, L> fixedDatumLabels = new HashMap<D, L>(); 
	
	/**
	 * @return a generic instance of some model that can be used
	 * when deserializing from an experiment configuration file
	 */
	protected abstract SupervisedModel<D, L> makeInstance();
	
	/**
	 * 
	 * @param name - Name for the object given on the current line
	 * @param reader - Buffered reader pointed to directly after the name
	 * @param datumTools
	 * @return true if a single line of extra model info has been deserialized.
	 * The extra info should be given below the first line of the model
	 * in curly brackets in a form similar to the serialized
	 * form of the experiments in ark.experiment (see documentation for the
	 * classes there).  Each line of extra info should contain
	 * a single assignment of some parameter to some value.  These parameter values
	 * can contain nested descriptions of objects that must also be deserialized, whereas
	 * the hyper-parameters in the first line of the model description
	 * cannot contain objects that must also be deserialized.  For each line
	 * of extra info, the deserialize method will make a separate call to 'deserializeExtraInfo',
	 * and 'deserializeExtraInfo' should use ark.util.SerializationUtil with the 
	 * given reader to deserialize the remainder of line after '[name]='.
	 * 
	 * @throws IOException
	 */
	protected abstract boolean deserializeExtraInfo(String name, BufferedReader reader, Datum.Tools<D, L> datumTools) throws IOException;
	
	/**
	 * 
	 * @param reader
	 * @param datumTools
	 * @return true if learned parameters of the model have been deserialized.  Deserialize will
	 * make a single call to this method during deserialization, and it should deserialize
	 * all of the parameters at once, assuming the reader's stream will end at the last line
	 * of parameters.
	 * 
	 * @throws IOException
	 */
	protected abstract boolean deserializeParameters(BufferedReader reader, Datum.Tools<D, L> datumTools) throws IOException;	
	
	/**
	 * 
	 * @param writer
	 * @return true if the learned parameters of the model have been serialized using
	 * writer.  
	 * @throws IOException
	 */
	protected abstract boolean serializeParameters(Writer writer) throws IOException;
	
	/**
	 * 
	 * @param writer
	 * @return true if the model's extra info has been serialized using writer
	 * @throws IOException
	 */
	protected abstract boolean serializeExtraInfo(Writer writer) throws IOException;
	
	/**
	 * @return the generic name of the model in the configuration files.  For
	 * model class SupervisedModel[X], the generic name should usually be X.
	 */
	public abstract String getGenericName();
	
	public abstract boolean train(FeaturizedDataSet<D, L> data, FeaturizedDataSet<D, L> testData, List<SupervisedModelEvaluation<D, L>> evaluations);
	
	/**
	 * @param data
	 * @return a map from datums to distributions over labels for the datums
	 */
	public abstract Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data);
	
	public boolean setLabelMapping(LabelMapping<L> labelMapping) {
		this.labelMapping = labelMapping;
		return true;
	}
	
	public boolean fixDatumLabels(Map<D, L> fixedDatumLabels) {
		this.fixedDatumLabels = fixedDatumLabels;
		return true;
	}
	
	public Set<L> getValidLabels() {
		return this.validLabels;
	}
	
	public LabelMapping<L> getLabelMapping() {
		return this.labelMapping;
	}
	
	public L mapValidLabel(L label) {
		if (label == null)
			return null;
		if (this.labelMapping != null)
			label = this.labelMapping.map(label);
		if (this.validLabels.contains(label))
			return label;
		else
			return null;
	}
	
	public SupervisedModel<D, L> makeInstance(Set<L> validLabels, LabelMapping<L> labelMapping) {
		SupervisedModel<D, L> instance = makeInstance();
		
		instance.validLabels = validLabels;
		instance.labelMapping = labelMapping;
		
		return instance;
	}
	
	public String getReferenceName() {
		return this.referenceName;
	}
	
	public boolean deserialize(BufferedReader reader, boolean readGenericName, boolean readParameters, Datum.Tools<D, L> datumTools, String referenceName) throws IOException {		
		if (readGenericName && SerializationUtil.deserializeGenericName(reader) == null)
			return false;
		
		Map<String, String> parameters = SerializationUtil.deserializeArguments(reader);
		if (parameters != null)
			for (Entry<String, String> entry : parameters.entrySet()) {
				String parameterValue = entry.getValue();
				for (Entry<String, String> envEntry : datumTools.getDataTools().getParameterEnvironment().entrySet())
					parameterValue = parameterValue.replace("${" + envEntry.getKey() + "}", envEntry.getValue());
				
				if (!this.setParameterValue(entry.getKey(), parameterValue, datumTools))
					return false;
			}
		
		reader.readLine(); // Read "{"
		
		String assignmentLeft = null;
		while ((assignmentLeft = SerializationUtil.deserializeAssignmentLeft(reader)) != null) {
			if (assignmentLeft.equals("labelMapping"))
				this.labelMapping = datumTools.getLabelMapping(SerializationUtil.deserializeAssignmentRight(reader));
			else if (assignmentLeft.equals("validLabels")) {
				this.validLabels = new HashSet<L>();
				List<String> validLabelStrs = SerializationUtil.deserializeList(reader);
				for (String validLabelStr : validLabelStrs)
					this.validLabels.add(datumTools.labelFromString(validLabelStr));
			} else {
				if (!deserializeExtraInfo(assignmentLeft, reader, datumTools))
					return false;
			}
		}
		
		if (readParameters)
			deserializeParameters(reader, datumTools);	
	
		return true;
	}
	
	public boolean serialize(Writer writer, boolean includeReferenceName) throws IOException {
		if (includeReferenceName) {
			writer.write("model" + (this.referenceName == null ? "" : "_" + this.referenceName) + "=");
		}
		
		writer.write(toString(false));
		writer.write("\n{\n");
		if (this.labelMapping != null)
			writer.write("\tlabelMapping=" + this.labelMapping.toString() +"\n");
		writer.write("\tvalidLabels=");
		if (!SerializationUtil.serializeList(this.validLabels, writer))
			return false;
		writer.write("\n");
		
		if (!serializeExtraInfo(writer))
			return false;
		
		writer.write("}\n");
		
		if (!serializeParameters(writer))
			return false;
		
		return true;
	}
	
	public String toString(boolean withParameters) {
		if (withParameters) {
			StringWriter stringWriter = new StringWriter();
			try {
				if (serialize(stringWriter, withParameters))
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
		return toString(true);
	}
	
	public boolean fromString(String str, Datum.Tools<D, L> datumTools) {
		try {
			return deserialize(new BufferedReader(new StringReader(str)), true, false,  datumTools, null);
		} catch (IOException e) {
			
		}
		return true;
	}
	
	public  <D1 extends Datum<L1>, L1> SupervisedModel<D1, L1> clone(Datum.Tools<D1, L1> datumTools) {
		return clone(datumTools, null, true);
	}
	
	@SuppressWarnings("unchecked")
	public <D1 extends Datum<L1>, L1> SupervisedModel<D1, L1> clone(Datum.Tools<D1, L1> datumTools, Map<String, String> environment, boolean copyLabelObjects) {
		SupervisedModel<D1, L1> clone = datumTools.makeModelInstance(getGenericName(), true);
		
		if (copyLabelObjects) {
			clone.validLabels = (Set<L1>)this.validLabels;
			clone.labelMapping = (Datum.Tools.LabelMapping<L1>)this.labelMapping;
			clone.fixedDatumLabels = (Map<D1, L1>)this.fixedDatumLabels;
		}
		
		String[] parameterNames = getParameterNames();
		for (int i = 0; i < parameterNames.length; i++) {
			String parameterValue = getParameterValue(parameterNames[i]);
			if (parameterValue != null) {
				if (environment != null)
					for (Entry<String, String> entry : environment.entrySet())
						parameterValue = parameterValue.replace("--" + entry.getKey() + "--", entry.getValue());
			}
			
			clone.setParameterValue(parameterNames[i], parameterValue, datumTools);
		}
		
		clone.referenceName = this.referenceName;
		
		return clone;
	}
	
	public Map<D, L> classify(FeaturizedDataSet<D, L> data) {
		Map<D, L> classifiedData = new HashMap<D, L>();
		Map<D, Map<L, Double>> posterior = posterior(data);
	
		for (Entry<D, Map<L, Double>> datumPosterior : posterior.entrySet()) {
			if (this.fixedDatumLabels.containsKey(datumPosterior.getKey())) {
				classifiedData.put(datumPosterior.getKey(), this.fixedDatumLabels.get(datumPosterior.getKey()));
				continue;
			}
			
			Map<L, Double> p = datumPosterior.getValue();
			double max = Double.NEGATIVE_INFINITY;
			L argMax = null;
			for (Entry<L, Double> entry : p.entrySet()) {
				if (entry.getValue() > max) {
					max = entry.getValue();
					argMax = entry.getKey();
				}
			}
			classifiedData.put(datumPosterior.getKey(), argMax);
		}
	
		return classifiedData;
	}
	
	public boolean setHyperParameterValues(Map<String, String> values, Datum.Tools<D, L> datumTools) {
		for (Entry<String, String> entry : values.entrySet()) {
			if (!setParameterValue(entry.getKey(), entry.getValue(), datumTools))
				return false;
		}
		
		return true;
	}
	
	public boolean setLabels(Set<L> validLabels, LabelMapping<L> labelMapping) {
		this.validLabels = validLabels;
		this.labelMapping = labelMapping;
		return true;
	}
	
	public static <D extends Datum<L>, L> SupervisedModel<D, L> deserialize(BufferedReader reader, boolean readParameters, Datum.Tools<D, L> datumTools) throws IOException {		
		String assignmentLeft = SerializationUtil.deserializeAssignmentLeft(reader);
		
		String[] nameParts = assignmentLeft.split("_");
		String referenceName = null;
		if (nameParts.length > 1)
			referenceName = nameParts[1];
			
		String modelName = SerializationUtil.deserializeGenericName(reader);
		SupervisedModel<D, L> model = datumTools.makeModelInstance(modelName);
		if (!model.deserialize(reader, false, readParameters, datumTools, referenceName))
			return null;

		return model;
	}
}
