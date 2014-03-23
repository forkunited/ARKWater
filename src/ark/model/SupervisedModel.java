package ark.model;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelMapping;
import ark.data.feature.FeaturizedDataSet;
import ark.util.Pair;
import ark.util.SerializationUtil;

public abstract class SupervisedModel<D extends Datum<L>, L> {
	protected Set<L> validLabels;
	protected LabelMapping<L> labelMapping;
	
	protected abstract String[] getHyperParameterNames();
	protected abstract SupervisedModel<D, L> makeInstance();
	protected abstract boolean deserializeExtraInfo(Reader reader, Datum.Tools<D, L> datumTools);
	protected abstract boolean deserializeParameters(Reader reader, Datum.Tools<D, L> datumTools);	
	protected abstract boolean serializeParameters(Writer writer);
	protected abstract boolean serializeExtraInfo(Writer writer);
	
	public abstract String getGenericName();
	public abstract String getHyperParameterValue(String parameter);
	public abstract boolean setHyperParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools);

	public abstract boolean train(FeaturizedDataSet<D, L> data);
	public abstract Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data);
	
	public SupervisedModel<D, L> makeInstance(Set<L> validLabels, LabelMapping<L> labelMapping) {
		SupervisedModel<D, L> instance = makeInstance();
		
		instance.validLabels = validLabels;
		instance.labelMapping = labelMapping;
		
		return instance;
	}
	
	public boolean deserialize(Reader reader, boolean readGenericName, boolean readParameters, Datum.Tools<D, L> datumTools) throws IOException {
		BufferedReader bufferedReader = new BufferedReader(reader);
		
		if (readGenericName && SerializationUtil.deserializeGenericName(bufferedReader) == null)
			return false;
		
		Map<String, String> parameters = SerializationUtil.deserializeArguments(bufferedReader);
		for (Entry<String, String> entry : parameters.entrySet())
			this.setHyperParameterValue(entry.getKey(), entry.getValue(), datumTools);
		
		bufferedReader.readLine(); // Read "{"
		
		String line = null;
		while (!(line = bufferedReader.readLine()).contains("}")) {
			Pair<String, String> assignment = SerializationUtil.deserializeAssignment(new StringReader(line));
			if (assignment.getFirst().equals("labelMapping"))
				this.labelMapping = datumTools.getLabelMapping(assignment.getSecond());
			else if (assignment.getFirst().equals("validLabels")) {
				this.validLabels = new TreeSet<L>();
				List<String> validLabelStrs = SerializationUtil.deserializeList(new StringReader(assignment.getSecond()));
				for (String validLabelStr : validLabelStrs)
					this.validLabels.add(datumTools.labelFromString(validLabelStr));
			} else {
				if (!deserializeExtraInfo(reader, datumTools))
					return false;
			}
		}
		
		if (readParameters)
			deserializeParameters(bufferedReader, datumTools);	
	
		return true;
	}
	
	public boolean serialize(Writer writer) throws IOException {
		writer.write(toString(false));
		writer.write("\n{");
		if (this.labelMapping != null)
			writer.write("\tlabelMapping=" + this.labelMapping.toString() +"\n");
		writer.write("validLabels=" + SerializationUtil.serializeList(this.validLabels, writer) + "\n");
		
		if (!serializeExtraInfo(writer))
			return false;
		
		writer.write("}\n");
		
		if (!serializeParameters(writer))
			return false;
		
		return true;
	}
	
	public String toString(boolean withParameters) {
		if (withParameters) {
			BufferedWriter stringWriter = new BufferedWriter(new StringWriter());
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
			String[] parameterNames = getHyperParameterNames();
			for (int i = 0; i < parameterNames.length; i++)
				parameters.put(parameterNames[i], getHyperParameterValue(parameterNames[i]));
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
			return deserialize(new BufferedReader(new StringReader(str)), true, false,  datumTools);
		} catch (IOException e) {
			
		}
		return true;
	}
	
	public SupervisedModel<D, L> clone(Datum.Tools<D, L> datumTools) {
		SupervisedModel<D, L> clone = makeInstance();
		String[] parameterNames = getHyperParameterNames();
		for (int i = 0; i < parameterNames.length; i++)
			clone.setHyperParameterValue(parameterNames[i], getHyperParameterValue(parameterNames[i]), datumTools);
		return clone;
	}
	
	public Map<D, L> classify(FeaturizedDataSet<D, L> data) {
		Map<D, L> classifiedData = new HashMap<D, L>();
		Map<D, Map<L, Double>> posterior = posterior(data);
	
		for (Entry<D, Map<L, Double>> datumPosterior : posterior.entrySet()) {
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
	
	public Set<L> getValidLabels() {
		return this.validLabels;
	}
	
	public LabelMapping<L> getLabelMapping() {
		return this.labelMapping;
	}
	
	public boolean setHyperParameterValues(Map<String, String> values, Datum.Tools<D, L> datumTools) {
		for (Entry<String, String> entry : values.entrySet()) {
			if (!setHyperParameterValue(entry.getKey(), entry.getValue(), datumTools))
				return false;
		}
		
		return true;
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
}
