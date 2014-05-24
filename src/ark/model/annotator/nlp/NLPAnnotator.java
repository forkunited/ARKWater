package ark.model.annotator.nlp;

import ark.data.annotation.Document;
import ark.data.annotation.Language;
import ark.data.annotation.nlp.ConstituencyParse;
import ark.data.annotation.nlp.DependencyParse;
import ark.data.annotation.nlp.PoSTag;

public abstract class NLPAnnotator {
	protected Language language;
	protected String text;
	
	protected boolean disabledPoSTags;
	protected boolean disabledDependencyParses;
	protected boolean disabledConstituencyParses;
	
	public abstract boolean setLanguage(Language language);
	public abstract boolean setText(String text);
	public abstract String toString();
	
	public abstract String[][] makeTokens();
	protected abstract PoSTag[][] makePoSTagsInternal();
	protected abstract DependencyParse[] makeDependencyParsesInternal(Document document, int sentenceIndexOffset);
	protected abstract ConstituencyParse[] makeConstituencyParsesInternal(Document document, int sentenceIndexOffset);
	
	public void disablePoSTags() {
		this.disabledPoSTags = true;
	}
	
	public void disableDependencyParses() {
		this.disabledDependencyParses = true;
	}
	
	public void disableConstituencyParses() {
		this.disabledConstituencyParses = true;
	}
	
	public PoSTag[][] makePoSTags() {
		if (this.disabledPoSTags)
			return new PoSTag[0][];
		return makePoSTagsInternal();
	}
	
	public DependencyParse[] makeDependencyParses(Document document, int sentenceIndexOffset) {
		if (this.disabledDependencyParses)
			return new DependencyParse[0];
		return makeDependencyParsesInternal(document, sentenceIndexOffset);
	}
	
	public ConstituencyParse[] makeConstituencyParses(Document document, int sentenceIndexOffset) {
		if (this.disabledConstituencyParses)
			return new ConstituencyParse[0];
		return makeConstituencyParsesInternal(document, sentenceIndexOffset);
	}
	
	public DependencyParse[] makeDependencyParses() {
		return makeDependencyParses(null, 0);
	}
	
	public ConstituencyParse[] makeConstituencyParses() {
		return makeConstituencyParses(null, 0);
	}
}