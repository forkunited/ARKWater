package ark.model;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.DataTools;
import ark.data.annotation.Datum;
import ark.data.feature.Feature;
import ark.data.feature.FeaturizedDataSet;
import ark.util.FileUtil;
import ark.util.SerializationUtil;

public abstract class SupervisedModel<D extends Datum<L>, L> {
	protected String modelPath;
	protected List<L> validLabels;
	
	protected abstract String[] getHyperParameterNames();
	protected abstract SupervisedModel<D, L> makeInstance();
	protected abstract boolean deserializeParameters(BufferedReader reader, DataTools dataTools, Datum.Tools<D, L> datumTools);	
	
	public abstract String getGenericName();
	public abstract String getHyperParameterValue(String parameter);
	public abstract boolean setHyperParameterValue(String parameter, String parameterValue, DataTools dataTools, Datum.Tools<D, L> datumTools);

	public abstract boolean train(FeaturizedDataSet<D, L> data, String outputPath);
	public abstract Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data);
	
	
	public boolean deserialize(BufferedReader reader, boolean readGenericName, DataTools dataTools, Datum.Tools<D, L> datumTools) throws IOException {
		int cInt = -1;
		char c = 0;
		if (readGenericName) {
			cInt = reader.read();
			c = (char)cInt;
			while (cInt != -1 && c != '(') {
				cInt = reader.read();
				c = (char)cInt;
			}
			
			if (cInt == -1)
				return false;
		} // FIXME: Put in serialization util.  Add serialize, toString, fromString methods.
		
		Map<String, String> parameters = SerializationUtil.deserializeArguments(reader);
		for (Entry<String, String> entry : parameters.entrySet())
			this.setHyperParameterValue(entry.getKey(), entry.getValue(), dataTools, datumTools);
		
		reader.readLine();
		
		deserializeParameters(reader, dataTools, datumTools);	
		
		return true;
	}
	
	public SupervisedModel<D, L> clone(DataTools dataTools, Datum.Tools<D, L> datumTools) {
		SupervisedModel<D, L> clone = makeInstance();
		String[] parameterNames = getHyperParameterNames();
		for (int i = 0; i < parameterNames.length; i++)
			clone.setHyperParameterValue(parameterNames[i], getHyperParameterValue(parameterNames[i]), dataTools, datumTools);
		return clone;
	}
	
	public List<L> getValidLabels() {
		return this.validLabels;
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
	
	public boolean setParameterValues(Map<String, String> values, DataTools dataTools, Datum.Tools<D, L> datumTools) {
		for (Entry<String, String> entry : values.entrySet()) {
			if (!setHyperParameterValue(entry.getKey(), entry.getValue(), dataTools, datumTools))
				return false;
		}
		
		return true;
	}
}
