package ark.model.evaluation.metric;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.annotation.Datum;
import ark.util.Pair;
import ark.util.SerializationUtil;

public abstract class ClassificationEvaluation<D extends Datum<L>, L> {
	public abstract double compute(Collection<Pair<L, L>> actualAndPredicted);
	public abstract String getGenericName();
	
	protected abstract String[] getParameterNames();
	protected abstract String getParameterValue(String parameter);
	protected abstract boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools);
	protected abstract ClassificationEvaluation<D, L> makeInstance();
	
	public ClassificationEvaluation<D, L> clone(Datum.Tools<D, L> datumTools) {
		return clone(datumTools, null);
	}
	
	public ClassificationEvaluation<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		ClassificationEvaluation<D, L> clone = makeInstance();
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
			for (Entry<String, String> entry : parameters.entrySet())
				setParameterValue(entry.getKey(), entry.getValue(), datumTools);

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
