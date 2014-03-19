package ark.data.feature;

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.util.SerializationUtil;

public abstract class Feature<D extends Datum<L>, L> {
	public abstract boolean init(List<D> data, boolean append);
	public abstract Map<Integer, Double> computeVector(D datum);
	
	public abstract String getGenericName();
	public abstract String getGenericShortName();
	
	protected abstract Feature<D, L> makeInstance();
	protected abstract String getVocabularyTerm(int index); 
	protected abstract int getVocabularySize();
	protected abstract String[] getParameterNames();
	protected abstract String[] getParameterShortNames();
	protected abstract String getParameterValue(String parameter);
	protected abstract boolean setParameterValue(String parameter, String parameterValue);
	
	public Map<Integer, String> getSpecificShortNamesForIndices(Iterable<Integer> indices) {
		String prefix = getSpecificShortNamePrefix();
		Map<Integer, String> specificShortNames = new HashMap<Integer, String>();
		for (Integer index : indices) {
			specificShortNames.put(index, prefix + getVocabularyTerm(index));
		}
		
		return specificShortNames;
	}
	
	public List<String> getSpecificShortNames() {
		String prefix = getSpecificShortNamePrefix();
		int vocabularySize = getVocabularySize();
		List<String> specificShortNames = new ArrayList<String>(vocabularySize);
		for (int i = 0; i < vocabularySize; i++)
			specificShortNames.add(prefix + getVocabularyTerm(i));
		
		return specificShortNames;
	}
	
	public boolean init(List<D> data) {
		return init(data, false);
	}
	
	public Feature<D, L> clone() {
		Feature<D, L> clone = makeInstance();
		String[] parameters = getParameterShortNames();
		for (int i = 0; i < parameters.length; i++)
			clone.setParameterValue(parameters[i], getParameterValue(parameters[i]));
		return clone;
	}

	public boolean deserialize(Reader reader) {
		/* FIXME */
		return false;
	}
	
	public boolean serialize(Writer writer) throws IOException {
		int vocabularySize = getVocabularySize();
		writer.write(toString(false));
		if (vocabularySize > 0)
			writer.write("\t");
		
		for (int i = 0; i < vocabularySize; i++) {
			//String assignmentStr = SerializationUtil.serializeAssignment(new Pair<String, Integer>(getVocabularyTerm(i), i));
			/* FIXME */
		}
		
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
			String parametersStr = SerializationUtil.serializeArguments(parameters);
			return genericName + "(" + parametersStr + ")";
		}
	}
	
	public String toString() {
		return toString(false);
	}
	
	
	public boolean fromString(String str) {
		return deserialize(new StringReader(str));
	}
	
	protected String getSpecificShortNamePrefix() {
		StringBuilder shortNamePrefixBuilder = new StringBuilder();
		String genericName = getGenericShortName();
		String[] parameters = getParameterShortNames();
		
		shortNamePrefixBuilder = shortNamePrefixBuilder.append(genericName);
		for (int i = 0; i < parameters.length; i++)
			shortNamePrefixBuilder = shortNamePrefixBuilder.append(parameters[i])
														.append("-")
														.append(getParameterValue(parameters[i]))
														.append("_");
		
		return shortNamePrefixBuilder.toString();
	}
}
