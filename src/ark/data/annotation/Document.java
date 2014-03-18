package ark.data.annotation;

import java.util.List;

import ark.data.annotation.nlp.PoSTag;
import ark.data.annotation.nlp.TypedDependency;

public abstract class Document {
	protected String name;
	protected Language language;
	protected String nlpAnnotator;
	
	public String getName() {
		return this.name;
	}
	
	public Language getLanguage() {
		return this.language;
	}
	
	public String getNLPAnnotator() {
		return this.nlpAnnotator;
	}
	
	public abstract int getSentenceCount();
	public abstract int getSentenceTokenCount(int sentenceIndex);
	public abstract String getText();
	public abstract String getSentence(int sentenceIndex);
	public abstract String getToken(int sentenceIndex, int tokenIndex);
	public abstract PoSTag getPoSTag(int sentenceIndex, int tokenIndex);
	public abstract List<TypedDependency> getParentDependencies(int sentenceIndex, int tokenIndex);
	public abstract List<TypedDependency> getChildDependencies(int sentenceIndex, int tokenIndex);
	public abstract List<TypedDependency> getDependencies(int sentenceIndex);
}
