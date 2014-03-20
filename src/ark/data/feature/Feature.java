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
import java.util.Map.Entry;

import ark.data.DataTools;
import ark.data.annotation.DataSet;
import ark.data.annotation.Datum;
import ark.util.Pair;
import ark.util.SerializationUtil;

public abstract class Feature<D extends Datum<L>, L> {
	public abstract boolean init(DataSet<D, L> dataSet);
	public abstract Map<Integer, Double> computeVector(D datum);
	public abstract String getGenericName();
	public abstract int getVocabularySize();

	protected abstract String getVocabularyTerm(int index); 
	protected abstract boolean setVocabularyTerm(int index, String term);
	
	protected abstract String[] getParameterNames();
	protected abstract String getParameterValue(String parameter);
	protected abstract boolean setParameterValue(String parameter, String parameterValue, DataTools dataTools, Datum.Tools<D, L> datumTools);
	protected abstract Feature<D, L> makeInstance();
	
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
		for (int i = 0; i < vocabularySize; i++) {
			String vocabularyTerm = getVocabularyTerm(i);
			specificShortNames.add(prefix + ((vocabularyTerm == null) ? "" : vocabularyTerm));
		}
		
		return specificShortNames;
	}
	
	public Feature<D, L> clone(DataTools dataTools, Datum.Tools<D, L> datumTools) {
		Feature<D, L> clone = makeInstance();
		String[] parameterNames = getParameterNames();
		for (int i = 0; i < parameterNames.length; i++)
			clone.setParameterValue(parameterNames[i], getParameterValue(parameterNames[i]), dataTools, datumTools);
		return clone;
	}

	public boolean deserialize(Reader reader, boolean readGenericName, DataTools dataTools, Datum.Tools<D, L> datumTools) throws IOException {
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
		}
		
		Map<String, String> parameters = SerializationUtil.deserializeArguments(reader);
		for (Entry<String, String> entry : parameters.entrySet())
			this.setParameterValue(entry.getKey(), entry.getValue(), dataTools, datumTools);
		
		Pair<String, String> assignment = null;		
		do {
			assignment = SerializationUtil.deserializeAssignment(reader);
			if (assignment != null)
				if (!setVocabularyTerm(Integer.valueOf(assignment.getSecond()), assignment.getFirst()))
					return false;
		} while(assignment != null);

		return true;
	}
	
	public boolean serialize(Writer writer) throws IOException {
		int vocabularySize = getVocabularySize();
		writer.write(toString(false));
		writer.write("\t");
		
		for (int i = 0; i < vocabularySize; i++) {
			String vocabularyTerm = getVocabularyTerm(i);
			if (vocabularyTerm == null)
				continue;
			Pair<String, Integer> v = new Pair<String, Integer>(vocabularyTerm, i);
			if (!SerializationUtil.serializeAssignment(v, writer))
				return false;
			if (i != vocabularySize - 1)
				writer.write(",");
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
	
	
	public boolean fromString(String str, DataTools dataTools, Datum.Tools<D, L> datumTools) {
		try {
			return deserialize(new StringReader(str), true, dataTools, datumTools);
		} catch (IOException e) {
			
		}
		return true;
	}
	
	protected String getSpecificShortNamePrefix() {
		StringBuilder shortNamePrefixBuilder = new StringBuilder();
		String genericName = shortenName(getGenericName());
		String[] parameterNames = getParameterNames();
		
		shortNamePrefixBuilder = shortNamePrefixBuilder.append(genericName);
		for (int i = 0; i < parameterNames.length; i++)
			shortNamePrefixBuilder = shortNamePrefixBuilder.append(shortenName(parameterNames[i]))
														.append("-")
														.append(getParameterValue(parameterNames[i]))
														.append("_");
		
		return shortNamePrefixBuilder.toString();
	}
	
	private String shortenName(String name) {
		if (name.length() == 0)
			return name;
		
		StringBuilder shortenedName = new StringBuilder();
		shortenedName.append(name.charAt(0));
		
		int curWordSize = 0;
		for (int i = 1; i < name.length(); i++) {
			if (Character.isUpperCase(name.charAt(i))) {
				shortenedName.append(name.charAt(i));
				curWordSize = 1;
			} else if (curWordSize <= 4) {
				shortenedName.append(name.charAt(i));
				curWordSize++;
			}
		}
		
		return shortenedName.toString();
	}
}
