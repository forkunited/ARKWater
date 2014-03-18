package ark.model.annotator.nlp;

import ark.data.annotation.Language;
import ark.data.annotation.nlp.PoSTag;
import ark.data.annotation.nlp.TypedDependency;

public abstract class NLPAnnotator {
	protected Language language;
	protected String text;
	
	public abstract boolean setLanguage(Language language);
	public abstract boolean setText(String text);
	public abstract String toString();
	
	public abstract String[][] makeTokens();
	public abstract PoSTag[][] makePoSTags();
	public abstract TypedDependency[][] makeDependencies();
}