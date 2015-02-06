package ark.data.feature;

import java.io.BufferedReader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.util.BidirectionalLookupTable;
import ark.util.CounterTable;
import ark.util.ThreadMapper;

public class FeatureStringForm<D extends Datum<L>, L> extends Feature<D, L> {
	protected BidirectionalLookupTable<String, Integer> vocabulary;
	protected Datum.Tools.StringExtractor<D, L> stringExtractor;
	protected int minFeatureOccurrence;
	protected String[] parameterNames = { "stringExtractor", "minFeatureOccurrence" };
	
	@Override
	public boolean init(FeaturizedDataSet<D, L> dataSet) {
		final CounterTable<String> counter = new CounterTable<String>();
		dataSet.map(new ThreadMapper.Fn<D, Boolean>() {
			@Override
			public Boolean apply(D datum) {
				List<String> forms = computeForms(datum);
				for (String form : forms)
					counter.incrementCount(form);

				return true;
			}
		});
		
		counter.removeCountsLessThan(this.minFeatureOccurrence);
		
		this.vocabulary = new BidirectionalLookupTable<String, Integer>(counter.buildIndex());
		
		return true;
	}
	
	
	@Override
	public Map<Integer, Double> computeVector(D datum) {
		List<String> forms = computeForms(datum);
		Map<Integer, Double> vector = new HashMap<Integer, Double>();
		
		for (String form : forms) {
			if (!this.vocabulary.containsKey(form))
				continue;
			vector.put(this.vocabulary.get(form), 1.0);
		}
		
		return vector;
	}

	
	protected List<String> computeForms(D datum) {
		String[] strs = this.stringExtractor.extract(datum);
		List<String> forms = new ArrayList<String>(strs.length);
		
		for (String str : strs) {
			StringBuilder form = new StringBuilder();
			
			char[] characters = str.toCharArray();
			char prevFormCharacter = '\0';
			for (char character : characters) {
				char formCharacter = '\0';
				if (Character.isAlphabetic(character)) {
					if (Character.isLowerCase(character)) {
						formCharacter = 'a';
					} else {
						formCharacter = 'A';
					}
				} else if (Character.isDigit(character)) {
					formCharacter = 'D';
				} else {
					formCharacter = 'S';
				}
				
				if (formCharacter != prevFormCharacter)
					form.append(formCharacter);
				prevFormCharacter = formCharacter;
			}
			
			forms.add(form.toString());
		}
		
		return forms;
	}
	
	@Override
	public String[] getParameterNames() {
		return this.parameterNames;
	}

	@Override
	public String getParameterValue(String parameter) {
		if (parameter.equals("minFeatureOccurrence"))
			return String.valueOf(this.minFeatureOccurrence);
		else if (parameter.equals("stringExtractor"))
			return (this.stringExtractor == null) ? "" : this.stringExtractor.toString();
		return null;
	}

	@Override
	public boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (parameter.equals("minFeatureOccurrence"))
			this.minFeatureOccurrence = Integer.valueOf(this.minFeatureOccurrence);
		else if (parameter.equals("stringExtractor"))
			this.stringExtractor = datumTools.getStringExtractor(parameterValue);
		else 
			return false;
		return true;
	}
	
	@Override
	public String getVocabularyTerm(int index) {
		if (this.vocabulary == null)
			return null;
		return this.vocabulary.reverseGet(index);
	}

	@Override
	protected boolean setVocabularyTerm(int index, String term) {
		if (this.vocabulary == null)
			return true;
		this.vocabulary.put(term, index);
		return true;
	}

	@Override
	public int getVocabularySize() {
		if (this.vocabulary == null)
			return 1;
		return this.vocabulary.size();
	}
	
	@Override
	protected <D1 extends Datum<L1>, L1> boolean cloneHelper(Feature<D1, L1> clone, boolean newObjects) {
		if (!newObjects) {
			FeatureGazetteer<D1,L1> cloneFeature = (FeatureGazetteer<D1, L1>)clone;
			cloneFeature.vocabulary = this.vocabulary;
		}
		
		return true;
	}
	
	@Override
	protected boolean serializeHelper(Writer writer) {
		return true;
	}
	
	@Override
	protected boolean deserializeHelper(BufferedReader writer) {
		return true;
	}


	@Override
	public String getGenericName() {
		return "StringForm";
	}


	@Override
	public Feature<D, L> makeInstance() {
		return new FeatureStringForm<D, L>();
	}	

}
