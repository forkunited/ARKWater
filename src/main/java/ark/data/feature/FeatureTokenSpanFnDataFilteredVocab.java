package ark.data.feature;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import ark.data.Context;
import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools.LabelIndicator;
import ark.data.feature.FeatureTokenSpanFnDataVocab.Scale;
import ark.parse.Assignment;
import ark.parse.AssignmentList;
import ark.parse.Obj;
import ark.util.BidirectionalLookupTable;

public class FeatureTokenSpanFnDataFilteredVocab<D extends Datum<L>, L> extends Feature<D, L> {
	protected BidirectionalLookupTable<String, Integer> vocabulary;
	protected Map<Integer, Double> idfs; // maps vocabulary term indices to idf values to use in tfidf scale function

	public enum Type {
		SUFFIX,
		PREFIX
	}
	
	protected FeatureTokenSpanFnDataVocabTrie<D, L> vocabFeature;
	protected String filter;
	protected Type type;
	protected String[] parameterNames = {"vocabFeature", "filter", "type"};
	
	
	public FeatureTokenSpanFnDataFilteredVocab() {
		
	}
	
	public FeatureTokenSpanFnDataFilteredVocab(Context<D, L> context) {
		this.context = context;
		this.vocabulary = new BidirectionalLookupTable<String, Integer>();
		this.idfs = new HashMap<Integer, Double>();
	}
	
	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public Obj getParameterValue(String parameter) {
		if (parameter.equals("vocabFeature"))
			return this.vocabFeature.toParse();
		else if (parameter.equals("filter"))
			return Obj.stringValue(this.filter);
		else if (parameter.equals("type"))
			return Obj.stringValue(this.type.toString());
		else
			return null;
	}

	@Override
	public boolean setParameterValue(String parameter, Obj parameterValue) {
		if (parameter.equals("vocabFeature"))
			this.vocabFeature = (FeatureTokenSpanFnDataVocabTrie<D, L>)this.context.getMatchFeature(parameterValue);
		else if (parameter.equals("filter"))
			this.filter = this.context.getMatchValue(parameterValue);
		else if (parameter.equals("type"))
			this.type = Type.valueOf(this.context.getMatchValue(parameterValue));
		else
			return false;
		return true;
	}

	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		Map<String, Double> vocab = (this.type == Type.SUFFIX) ? this.vocabFeature.getVocabularyTermsSuffixedBy(this.filter)
															   : this.vocabFeature.getVocabularyTermsPrefixedBy(this.filter);
		
		int i = 0;
		for (Entry<String, Double> entry : vocab.entrySet()) {
			this.vocabulary.put(entry.getKey(), i);
			this.idfs.put(i, entry.getValue());
			i++;
		}
		
		return true;
	}

	@Override
	public Map<Integer, Double> computeVector(D datum) {
		Map<String, Integer> gramsForDatum = this.vocabFeature.applyFnToDatum(datum);
		Scale scale = this.vocabFeature.getScale();
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		
		if (scale == Scale.INDICATOR) {
			for (String gram : gramsForDatum.keySet()) {
				if (this.vocabulary.containsKey(gram))
					vector.put(this.vocabulary.get(gram), 1.0);		
			}
		} else if (scale == Scale.NORMALIZED_LOG) {
			double norm = 0.0;
			for (Entry<String, Integer> entry : gramsForDatum.entrySet()) {
				if (!this.vocabulary.containsKey(entry.getKey()))
					continue;
				int index = this.vocabulary.get(entry.getKey());
				double value = Math.log(entry.getValue() + 1.0);
				norm += value*value;
				vector.put(index, value);
			}
			
			norm = Math.sqrt(norm);
			
			for (Entry<Integer, Double> entry : vector.entrySet()) {
				entry.setValue(entry.getValue()/norm);
			}
		} else if (scale == Scale.NORMALIZED_TFIDF) {
			double norm = 0.0;
			for (Entry<String, Integer> entry : gramsForDatum.entrySet()) {
				if (!this.vocabulary.containsKey(entry.getKey()))
					continue;
				int index = this.vocabulary.get(entry.getKey());
				double value = entry.getValue()*this.idfs.get(index);//Math.log(entry.getValue() + 1.0);
				norm += value*value;
				vector.put(index, value);
			}
			
			norm = Math.sqrt(norm);
			
			for (Entry<Integer, Double> entry : vector.entrySet()) {
				entry.setValue(entry.getValue()/norm);
			}
		}

		return vector;
	}

	@Override
	public int getVocabularySize() {
		return this.vocabulary.size();
	}

	@Override
	public String getVocabularyTerm(int index) {
		return this.vocabulary.reverseGet(index);
	}

	@Override
	protected boolean setVocabularyTerm(int index, String term) {
		this.vocabulary.put(term, index);
		return true;
	}

	@Override
	protected <T extends Datum<Boolean>> Feature<T, Boolean> makeBinaryHelper(
			Context<T, Boolean> context, LabelIndicator<L> labelIndicator,
			Feature<T, Boolean> binaryFeature) {
		
		FeatureTokenSpanFnDataFilteredVocab<T, Boolean> binaryFeatureTokenSpanFnDataFilteredVocab = (FeatureTokenSpanFnDataFilteredVocab<T, Boolean>)binaryFeature;
		
		binaryFeatureTokenSpanFnDataFilteredVocab.vocabulary = this.vocabulary;
		binaryFeatureTokenSpanFnDataFilteredVocab.idfs = this.idfs;
		
		return binaryFeatureTokenSpanFnDataFilteredVocab;
	}

	@Override
	protected boolean fromParseInternalHelper(AssignmentList internalAssignments) {
		if (internalAssignments == null)
			return true;
		if (!internalAssignments.contains("idfs"))
			return false;
		
		this.idfs = new HashMap<Integer, Double>();
		
		Obj.Array idfs = (Obj.Array)internalAssignments.get("idfs").getValue();
		for (int i = 0; i < idfs.size(); i++)
			this.idfs.put(i, Double.valueOf(idfs.get(i).getStr()));
		
		return true;
	}

	@Override
	protected AssignmentList toParseInternalHelper(
			AssignmentList internalAssignments) {
		if (this.vocabulary.size() == 0)
			return internalAssignments;
		
		Obj.Array idfs = Obj.array();
		for (int i = 0; i < this.vocabulary.size(); i++) {
			if (this.idfs.containsKey(i))
				idfs.add(Obj.stringValue(String.valueOf(this.idfs.get(i))));
			else
				idfs.add(Obj.stringValue("0.0"));
		}
		
		internalAssignments.add(Assignment.assignmentTyped(new ArrayList<String>(), Context.ARRAY_STR, "idfs", idfs));
		
		return internalAssignments;
	}

	@Override
	public Feature<D, L> makeInstance(Context<D, L> context) {
		return new FeatureTokenSpanFnDataFilteredVocab<D, L>(context);
	}

	@Override
	public String getGenericName() {
		return "TokenSpanFnDataFilteredVocab";
	}

}
