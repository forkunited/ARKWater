package ark.data.feature;

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
import ark.util.Pair;
import ark.util.SerializationUtil;

/**
 * Feature represents an abstract feature to be computed on data and
 * used in a model.
 * 
 * Implementations of particular features derive from the Feature class,
 * and the Feature class is primarily responsible for providing the
 * methods necessary for deserializing features from
 * configuration files.  The features are defined by strings in the 
 * configuration file of the form:
 * 
 * feature(_[featureReferenceName](_ignored))=[featureGenericName]([parameterName1]=[parameterValue1],...)
 * 
 * Where strings in parentheses are optional and strings in square brackets
 * are replaced by feature specific information.
 * 
 * A feature's computed values are generally vectors of real values.  
 * Each component of a feature's vector has a name, and the set of all names 
 * for components is the feature's 'vocabulary'.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public abstract class Feature<D extends Datum<L>, L> {
	// particular name by which the feature is referenced in configuration files
	private String referenceName; 
	// indicator of whether to ignore the feature in data sets so that it
	// isn't included in models
	private boolean ignored;
	
	/**
	 * @param dataSet
	 * @return true if the feature has been initialized for the dataSet
	 */
	public abstract boolean init(FeaturizedDataSet<D, L> dataSet);
	
	/**
	 * @param datum
	 * @return a sparse mapping from vector indices to values of the feature
	 * for the given datum.
	 */
	public abstract Map<Integer, Double> computeVector(D datum);
	
	/**
	 * @return the generic name of the feature in the configuration files.  For
	 * feature class Feature[X], the generic name should usually be X.
	 */
	public abstract String getGenericName();
	
	/**
	 * @return the length of the vector computed by this feature for each
	 * datum
	 */
	public abstract int getVocabularySize();
	
	/**
	 * 
	 * @param index
	 * @return the name of the component at the given index within vectors
	 * computed by this feature
	 * 
	 */
	public abstract String getVocabularyTerm(int index); 
	
	/**
	 * 
	 * @param index
	 * @param term
	 * @return true if the name of component at the given index has been set
	 * to the value of term.  This is used when deserializing features that 
	 * were previously computed and saved.
	 * 
	 */
	protected abstract boolean setVocabularyTerm(int index, String term);

	/**
	 * @return parameters of the feature that can be set through the experiment 
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
	 * @return a generic instance of the feature.  This is used when deserializing
	 * the parameters for the feature from a configuration file
	 */
	protected abstract Feature<D, L> makeInstance();
	
	/**
	 * @return a name by which this particular feature is referenced by other
	 * features in experiment configuration files.  This feature can be retrieved
	 * from a FeaturizedDataSet using this name.
	 * 
	 */
	public String getReferenceName() {
		return this.referenceName;
	}
	
	/**
	 * @return true if this feature should be ignored by models (it is only used for the
	 * computation of other features)
	 */
	public boolean isIgnored() {
		return this.ignored;
	}
	
	public Map<Integer, String> getSpecificShortNamesForIndices(Iterable<Integer> indices) {
		String prefix = getSpecificShortNamePrefix();
		Map<Integer, String> specificShortNames = new HashMap<Integer, String>();
		for (Integer index : indices) {
			specificShortNames.put(index, prefix + getVocabularyTerm(index));
		}
		
		return specificShortNames;
	}
	
	public Map<Integer, String> getVocabularyForIndices(Iterable<Integer> indices) {
		Map<Integer, String> vocabulary = new HashMap<Integer, String>();
		for (Integer index : indices) {
			vocabulary.put(index, getVocabularyTerm(index));
		}
		
		return vocabulary;
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
	
	public Feature<D, L> clone(Datum.Tools<D, L> datumTools) {
		return clone(datumTools, null);
	}
	
	public Feature<D, L> clone(Datum.Tools<D, L> datumTools, Map<String, String> environment) {
		Feature<D, L> clone = makeInstance();
		String[] parameterNames = getParameterNames();
		for (int i = 0; i < parameterNames.length; i++) {
			String parameterValue = getParameterValue(parameterNames[i]);
			if (environment != null && parameterValue != null) {
				for (Entry<String, String> entry : environment.entrySet())
					parameterValue = parameterValue.replace("${" + entry.getKey() + "}", entry.getValue());
			}
			clone.setParameterValue(parameterNames[i], parameterValue, datumTools);
		}
		
		clone.referenceName = this.referenceName;
		clone.ignored = this.ignored;
		
		return clone;
	}

	public boolean deserialize(BufferedReader reader, boolean readGenericName, boolean readVocabulary, Datum.Tools<D, L> datumTools, String referenceName, boolean ignored) throws IOException {		
		if (readGenericName && SerializationUtil.deserializeGenericName(reader) == null)
			return false;
		
		Map<String, String> parameters = SerializationUtil.deserializeArguments(reader);
		if (parameters != null)
			for (Entry<String, String> entry : parameters.entrySet())
				this.setParameterValue(entry.getKey(), entry.getValue(), datumTools);
		
		if (readVocabulary) {
			Map<String, String> vocabulary = SerializationUtil.deserializeArguments(reader);
			if (vocabulary == null)
				return true;
			
			for (Entry<String, String> entry : vocabulary.entrySet()) {
				if (!setVocabularyTerm(Integer.valueOf(entry.getValue()), entry.getKey()))
					return false;
			}
		}

		this.referenceName = referenceName;
		this.ignored = ignored;
		
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
	
	
	public boolean fromString(String str, Datum.Tools<D, L> datumTools) {
		try {
			return deserialize(new BufferedReader(new StringReader(str)), true, true, datumTools, null, false);
		} catch (IOException e) {
			
		}
		return true;
	}
	
	protected String getSpecificShortNamePrefix() {
		if (this.referenceName != null)
			return this.referenceName + "_";
		
		StringBuilder shortNamePrefixBuilder = new StringBuilder();
		String genericName = shortenName(getGenericName());
		String[] parameterNames = getParameterNames();
		
		shortNamePrefixBuilder = shortNamePrefixBuilder.append(genericName).append("_");
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
			} else if (curWordSize < 4) {
				shortenedName.append(name.charAt(i));
				curWordSize++;
			}
		}
		
		return shortenedName.toString();
	}
}
