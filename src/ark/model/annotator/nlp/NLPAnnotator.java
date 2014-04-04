package ark.model.annotator.nlp;

import ark.data.annotation.Language;
import ark.data.annotation.nlp.ConstituencyParse;
import ark.data.annotation.nlp.DependencyParse;
import ark.data.annotation.nlp.PoSTag;

public abstract class NLPAnnotator {
	protected Language language;
	protected String text;
	
	public abstract boolean setLanguage(Language language);
	public abstract boolean setText(String text);
	public abstract String toString();
	
	public abstract String[][] makeTokens();
	public abstract PoSTag[][] makePoSTags();
	public abstract DependencyParse[] makeDependencyParses();
	public abstract ConstituencyParse[] makeConstituencyParses();
}