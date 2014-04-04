package ark.data.annotation;

import java.util.ArrayList;
import java.util.List;

import ark.data.annotation.nlp.ConstituencyParse;
import ark.data.annotation.nlp.DependencyParse;
import ark.data.annotation.nlp.PoSTag;

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
	
	public List<String> getSentenceTokens(int sentenceIndex) {
		int sentenceTokenCount = getSentenceTokenCount(sentenceIndex);
		List<String> sentenceTokens = new ArrayList<String>(sentenceTokenCount);
		for (int i = 0; i < sentenceTokenCount; i++)
			sentenceTokens.add(getToken(sentenceIndex, i));
		return sentenceTokens;
	}
	
	public List<PoSTag> getSentencePoSTags(int sentenceIndex) {
		int sentenceTokenCount = getSentenceTokenCount(sentenceIndex);
		List<PoSTag> sentencePoSTags = new ArrayList<PoSTag>(sentenceTokenCount);
		for (int i = 0; i < sentenceTokenCount; i++)
			sentencePoSTags.add(getPoSTag(sentenceIndex, i));
		return sentencePoSTags;
	}
	
	public abstract int getSentenceCount();
	public abstract int getSentenceTokenCount(int sentenceIndex);
	public abstract String getText();
	public abstract String getSentence(int sentenceIndex);
	public abstract String getToken(int sentenceIndex, int tokenIndex);
	public abstract PoSTag getPoSTag(int sentenceIndex, int tokenIndex);
	public abstract ConstituencyParse getConstituencyParse(int sentenceIndex);
	public abstract DependencyParse getDependencyParse(int sentenceIndex);
}
