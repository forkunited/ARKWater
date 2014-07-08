package ark.data.feature;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;

/**
 * FeatureIdentity returns a vector D(d) for double
 * extractor D applied to datum d.
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public class FeatureIdentity<D extends Datum<L>, L> extends Feature<D, L> {
	protected Datum.Tools.DoubleExtractor<D, L> doubleExtractor;
	protected String[] parameterNames = {"doubleExtractor"};
	
	protected int vocabularySize;
	
	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		Iterator<D> dataIter = dataSet.iterator();
		if (!dataIter.hasNext())
			return false;
		
		D datum = dataSet.iterator().next();
		this.vocabularySize = this.doubleExtractor.extract(datum).length;
		
		return true;
	}

	@Override
	public Map<Integer, Double> computeVector(D datum) {
		double[] values = this.doubleExtractor.extract(datum);
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		for (int i = 0; i < this.vocabularySize; i++)
			if (values[i] != 0)
				vector.put(i, values[i]);
		return vector;
	}


	@Override
	public int getVocabularySize() {
		return this.vocabularySize;
	}


	@Override
	protected String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	protected String getParameterValue(String parameter) {
		if (parameter.equals("doubleExtractor"))
			return (this.doubleExtractor == null) ? null : this.doubleExtractor.toString();
		return null;
	}

	@Override
	protected boolean setParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		if (parameter.equals("doubleExtractor"))
			this.doubleExtractor = datumTools.getDoubleExtractor(parameterValue);
		else
			return false;
		return true;
	}

	@Override
	public String getVocabularyTerm(int index) {
		return String.valueOf(index);
	}

	@Override
	protected boolean setVocabularyTerm(int index, String term) {
		return true;
	}

	
	@Override
	public String getGenericName() {
		return "Identity";
	}
	
	@Override
	protected Feature<D, L> makeInstance() {
		return new FeatureIdentity<D, L>();
	}
}
