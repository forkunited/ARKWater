package ark.data.feature;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
import java.util.Map;
import ark.data.annotation.Datum;
import ark.data.annotation.nlp.TokenSpan;

public class FeatureTokenCount<D extends Datum<L>, L> extends Feature<D, L> {
	protected Datum.Tools.TokenSpanExtractor<D, L> tokenExtractor;
	protected int maxCount;
	protected String[] parameterNames = {"tokenExtractor", "maxCount"};
	
	public FeatureTokenCount() {
		
	}
	
	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		return true;
	}

	@Override
	public Map<Integer, Double> computeVector(D datum) {
		TokenSpan[] tokenSpans = this.tokenExtractor.extract(datum);
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		for (TokenSpan tokenSpan : tokenSpans) {
			int tokenCount = tokenSpan.getLength();
			if (tokenCount > this.maxCount)
				tokenCount = this.maxCount;
			
			vector.put(tokenCount, 1.0);
		}

		return vector;
	}

	@Override
	public String getVocabularyTerm(int index) {
		if (index > this.maxCount)
			return String.valueOf(this.maxCount);
		else
			return String.valueOf(index);
	}

	@Override
	protected boolean setVocabularyTerm(int index, String term) {
		return true;
	}

	@Override
	public int getVocabularySize() {
		return this.maxCount + 1;
	}

	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public String getParameterValue(String parameter) {
		if (parameter.equals("maxCount")) 
			return String.valueOf(this.maxCount);
		else if (parameter.equals("tokenExtractor"))
			return (this.tokenExtractor == null) ? null : this.tokenExtractor.toString();
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (parameter.equals("maxCount")) 
			this.maxCount = Integer.valueOf(parameterValue);
		else if (parameter.equals("tokenExtractor"))
			this.tokenExtractor = datumTools.getTokenSpanExtractor(parameterValue);
		else
			return false;
		return true;
	}
	
	@Override
	protected <D1 extends Datum<L1>, L1> boolean cloneHelper(Feature<D1, L1> clone, boolean newObjects) {
		return true;
	}
	
	@Override
	protected boolean serializeHelper(Writer writer) throws IOException {
		return true;
	}
	
	@Override
	protected boolean deserializeHelper(BufferedReader reader) throws IOException {
		return true;
	}

	@Override
	public String getGenericName() {
		return "TokenCount";
	}

	@Override
	public Feature<D, L> makeInstance() {
		return new FeatureTokenCount<D, L>();
	}
}
