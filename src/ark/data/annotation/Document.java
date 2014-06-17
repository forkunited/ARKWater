package ark.data.annotation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import net.sf.json.JSONObject;

import ark.data.annotation.nlp.ConstituencyParse;
import ark.data.annotation.nlp.DependencyParse;
import ark.data.annotation.nlp.PoSTag;
import ark.util.FileUtil;

public abstract class Document {
	protected String name;
	protected Language language;
	protected String nlpAnnotator;
	
	public Document() {
		
	}
	
	public Document(JSONObject json) {
		fromJSON(json);
	}
	
	public Document(String jsonPath) {
		BufferedReader r = FileUtil.getFileReader(jsonPath);
		String line = null;
		StringBuffer lines = new StringBuffer();
		try {
			while ((line = r.readLine()) != null) {
				lines.append(line).append("\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		fromJSON(JSONObject.fromObject(lines.toString()));
	}
	
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
	
	public boolean saveToJSONFile(String path) {
		try {
			BufferedWriter w = new BufferedWriter(new FileWriter(path));
			
			w.write(toJSON().toString());
			
			w.close();
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
		
		return true;
	}
	
	public abstract int getSentenceCount();
	public abstract int getSentenceTokenCount(int sentenceIndex);
	public abstract String getText();
	public abstract String getSentence(int sentenceIndex);
	public abstract String getToken(int sentenceIndex, int tokenIndex);
	public abstract PoSTag getPoSTag(int sentenceIndex, int tokenIndex);
	public abstract ConstituencyParse getConstituencyParse(int sentenceIndex);
	public abstract DependencyParse getDependencyParse(int sentenceIndex);
	public abstract JSONObject toJSON();
	public abstract Document makeInstanceFromJSONFile(String path);
	protected abstract boolean fromJSON(JSONObject json);
}
