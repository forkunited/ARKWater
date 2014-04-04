package ark.data.feature;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ark.data.annotation.Datum;
import ark.data.annotation.nlp.TokenSpan;
import ark.data.annotation.nlp.DependencyParse;

public class FeatureNGramDep<D extends Datum<L>, L> extends FeatureNGram<D, L> {
	public enum Mode {
		ParentsOnly,
		ChildrenOnly,
		ParentsAndChildren
	}
	
	private FeatureNGramDep.Mode mode;
	private boolean useRelationTypes;
	
	public FeatureNGramDep() {
		super();
		
		this.mode = Mode.ParentsAndChildren;
		this.useRelationTypes = true;
		
		this.parameterNames = Arrays.copyOf(this.parameterNames, this.parameterNames.length + 2);
		this.parameterNames[this.parameterNames.length - 2] = "mode";
		this.parameterNames[this.parameterNames.length - 1] = "useRelationTypes";
	}
	
	@Override
	protected Set<String> getNGramsForDatum(D datum) {
		TokenSpan[] tokenSpans = this.tokenExtractor.extract(datum);
		Set<String> retNgrams = new HashSet<String>();
		
		for (TokenSpan tokenSpan : tokenSpans) {
			List<String> tokens = tokenSpan.getDocument().getSentenceTokens(tokenSpan.getSentenceIndex());
			if (tokens == null)
				continue;
			int startIndex = tokenSpan.getStartTokenIndex();
			int endIndex = tokenSpan.getEndTokenIndex();
			DependencyParse dependencyParse = tokenSpan.getDocument().getDependencyParse(tokenSpan.getSentenceIndex());
			
			for (int i = startIndex; i < endIndex; i++) {
				if (this.mode == FeatureNGramDep.Mode.ChildrenOnly || this.mode == FeatureNGramDep.Mode.ParentsAndChildren) {
					List<DependencyParse.Dependency> dependencies = dependencyParse.getGovernedDependencies(i);
					for (DependencyParse.Dependency dependency : dependencies) {
						int depIndex = dependency.getDependentTokenIndex();
						if (depIndex <= tokens.size() - this.n && (depIndex < startIndex || depIndex >= endIndex)) {
							List<String> ngrams = getCleanNGrams(tokens, depIndex);
							for (String ngram : ngrams) {
								String retNgram = ngram + "_C";
								if (this.useRelationTypes)
									retNgram += "_" + ((dependency.getType() == null) ? "" : dependency.getType());
								retNgrams.add(retNgram);
							}
						}
					}
				}
				
				if (this.mode == FeatureNGramDep.Mode.ParentsOnly || this.mode == FeatureNGramDep.Mode.ParentsAndChildren) {
					List<DependencyParse.Dependency> dependencies = dependencyParse.getGoverningDependencies(i);
					for (DependencyParse.Dependency dependency : dependencies) {
						int govIndex = dependency.getGoverningTokenIndex();
						if (govIndex <= tokens.size() - this.n && (govIndex < startIndex || govIndex >= endIndex)) {
							List<String> ngrams = getCleanNGrams(tokens, govIndex);
							for (String ngram : ngrams) {
								String retNgram = ngram + "_P";
								if (this.useRelationTypes)
									retNgram += "_" + ((dependency.getType() == null) ? "" : dependency.getType());
								retNgrams.add(retNgram);
							}
						}
					}
				}
			}
		}
		
		return retNgrams;
	}

	@Override
	public String getGenericName() {
		return "NGramDep";
	}

	@Override
	protected Feature<D, L> makeInstance() {
		return new FeatureNGramDep<D, L>();
	}
	
	@Override
	protected String getParameterValue(String parameter) {
		String parameterValue = super.getParameterValue(parameter);
		if (parameterValue != null)
			return parameterValue;
		else if (parameter.equals("mode"))
			return this.mode == null ? null : this.mode.toString();
		else if (parameter.equals("useRelationTypes"))
			return String.valueOf(this.useRelationTypes);
		return null;
	}

	@Override
	protected boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools) {
		if (super.setParameterValue(parameter, parameterValue, datumTools))
			return true;
		else if (parameter.equals("mode"))
			this.mode = Mode.valueOf(parameterValue);
		else if (parameter.equals("useRelationTypes"))
			this.useRelationTypes = Boolean.valueOf(parameterValue);
		else
			return false;
		
		return true;
	}

}
