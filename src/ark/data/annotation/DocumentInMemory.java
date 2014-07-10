/**
 * Copyright 2014 Bill McDowell 
 *
 * This file is part of theMess (https://github.com/forkunited/theMess)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy 
 * of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the 
 * License for the specific language governing permissions and limitations 
 * under the License.
 */

package ark.data.annotation;

import net.sf.json.JSONArray;
import net.sf.json.JSONObject;

import ark.data.annotation.nlp.ConstituencyParse;
import ark.data.annotation.nlp.DependencyParse;
import ark.data.annotation.nlp.PoSTag;
import ark.model.annotator.nlp.NLPAnnotator;

/**
 * DocumentInMemory represents a text document with various 
 * NLP annotations (e.g. PoS tags, parses, etc) kept in 
 * memory.  
 * 
 * @author Bill McDowell
 * 
 */
public class DocumentInMemory extends Document {	
	protected String[][] tokens;
	protected PoSTag[][] posTags;
	protected DependencyParse[] dependencyParses; 
	protected ConstituencyParse[] constituencyParses;
	
	public DocumentInMemory() {
		
	}

	public DocumentInMemory(JSONObject json) {
		super(json);
	}
	
	public DocumentInMemory(String jsonPath) {
		super(jsonPath);
	}
	
	
	public DocumentInMemory(String name, String text, Language language, NLPAnnotator annotator) {
		this.name = name;
		this.language = language;
		this.nlpAnnotator = annotator.toString();
		
		annotator.setLanguage(language);
		annotator.setText(text);
		
		this.tokens = annotator.makeTokens();
		this.dependencyParses = annotator.makeDependencyParses(this, 0);
		this.constituencyParses = annotator.makeConstituencyParses(this, 0);
		this.posTags = annotator.makePoSTags();
	}
	
	public DocumentInMemory(String name, String[] sentences, Language language, NLPAnnotator annotator) {
		this.name = name;
		this.language = language;
		this.nlpAnnotator = annotator.toString();
		this.dependencyParses = new DependencyParse[sentences.length];
		this.constituencyParses = new ConstituencyParse[sentences.length];
		this.tokens = new String[sentences.length][];
		this.posTags = new PoSTag[sentences.length][];
		
		annotator.setLanguage(language);
		
		for (int i = 0; i < sentences.length; i++) {
			annotator.setText(sentences[i]);
			
			String[][] sentenceTokens = annotator.makeTokens();
			if (sentenceTokens.length > 1)
				throw new IllegalArgumentException("Input sentences are not actually sentences according to annotator...");
			else if (sentenceTokens.length == 0) {
				this.tokens[i] = new String[0];
				this.dependencyParses[i] = new DependencyParse(this, i);
				this.constituencyParses[i] = new ConstituencyParse(this, i);
				this.posTags[i] = new PoSTag[0];
				continue;
			}
			
			this.tokens[i] = new String[sentenceTokens[0].length];
			for (int j = 0; j < sentenceTokens[0].length; j++)
				this.tokens[i][j] = sentenceTokens[0][j];
						
			this.dependencyParses[i] = annotator.makeDependencyParses(this, i)[0];
			this.constituencyParses[i] = annotator.makeConstituencyParses(this, i)[0];
			
			PoSTag[][] sentencePoSTags = annotator.makePoSTags();
			this.posTags[i] = new PoSTag[sentencePoSTags[0].length];
			for (int j = 0; j < sentencePoSTags[0].length; j++)
				this.posTags[i][j] = sentencePoSTags[0][j];
			
		}
	}
	
	public int getSentenceCount() {
		return this.tokens.length;
	}
	
	public int getSentenceTokenCount(int sentenceIndex) {
		return this.tokens[sentenceIndex].length;
	}
	
	public String getText() {
		StringBuilder text = new StringBuilder();
		for (int i = 0; i < getSentenceCount(); i++)
			text = text.append(getSentence(i)).append(" ");
		return text.toString().trim();
	}
	
	public String getSentence(int sentenceIndex) {
		StringBuilder sentenceStr = new StringBuilder();
		
		for (int i = 0; i < this.tokens[sentenceIndex].length; i++) {
			sentenceStr = sentenceStr.append(this.tokens[sentenceIndex][i]).append(" ");
		}
		return sentenceStr.toString().trim();
	}
	
	public String getToken(int sentenceIndex, int tokenIndex) {
		if (tokenIndex < 0)
			return "ROOT";
		else 
			return this.tokens[sentenceIndex][tokenIndex];
	}
	
	public PoSTag getPoSTag(int sentenceIndex, int tokenIndex) {
		return this.posTags[sentenceIndex][tokenIndex];
	}
	
	@Override
	public ConstituencyParse getConstituencyParse(int sentenceIndex) {
		return this.constituencyParses[sentenceIndex];
	}

	@Override
	public DependencyParse getDependencyParse(int sentenceIndex) {
		return this.dependencyParses[sentenceIndex];
	}
	
	public boolean setPoSTags(PoSTag[][] posTags) {
		if (this.tokens.length != posTags.length)
			return false;
		this.posTags = new PoSTag[this.tokens.length][];
		for (int i = 0; i < posTags.length; i++) {
			if (this.tokens[i].length != posTags[i].length)
				return false;
			this.posTags[i] = new PoSTag[posTags[i].length];
			for (int j = 0; j < posTags[i].length; j++) {
				this.posTags[i][j] = posTags[i][j];
			}
		}
		
		return true;
	}
	
	public boolean setDependencyParses(DependencyParse[] dependencyParses) {
		this.dependencyParses = new DependencyParse[this.tokens.length];
		for (int i = 0; i < this.dependencyParses.length; i++)
			this.dependencyParses[i] = dependencyParses[i].clone(this);
		
		return true;
	}
	
	public boolean setConstituencyParses(ConstituencyParse[] constituencyParses) {
		this.constituencyParses = new ConstituencyParse[this.tokens.length];
		for (int i = 0; i < this.constituencyParses.length; i++)
			this.constituencyParses[i] = constituencyParses[i].clone(this);
	
		return true;
	}
	
	@Override
	public JSONObject toJSON() {
		JSONObject json = new JSONObject();
		JSONArray sentencesJson = new JSONArray();
		
		json.put("name", this.name);
		json.put("language", this.language.toString());
		json.put("nlpAnnotator", this.nlpAnnotator);
		
		int sentenceCount = getSentenceCount();
		for (int i = 0; i < sentenceCount; i++) {
			int tokenCount = getSentenceTokenCount(i);
			JSONObject sentenceJson = new JSONObject();
			sentenceJson.put("sentence", getSentence(i));
			
			JSONArray tokensJson = new JSONArray();
			JSONArray posTagsJson = new JSONArray();
			
			for (int j = 0; j < tokenCount; j++) {
				tokensJson.add(getToken(i, j));
				if (this.posTags.length > 0) {
					PoSTag posTag = getPoSTag(i, j);
					if (posTag != null)
						posTagsJson.add(posTag.toString());	
				}
			}
			
			sentenceJson.put("tokens", tokensJson);
			if (this.posTags.length > 0)
				sentenceJson.put("posTags", posTagsJson);
			if (this.dependencyParses.length > 0)
				sentenceJson.put("dependencyParse", getDependencyParse(i).toString());
			if (this.constituencyParses.length > 0)
				sentenceJson.put("constituencyParse", getConstituencyParse(i).toString());
			
			sentencesJson.add(sentenceJson);
		}
		json.put("sentences", sentencesJson);
		
		return json;
	}
	
	@Override
	protected boolean fromJSON(JSONObject json) {
		this.name = json.getString("name");
		this.language = Language.valueOf(json.getString("language"));
		
		if (json.has("nlpAnnotator"))
			this.nlpAnnotator = json.getString("nlpAnnotator");
		
		JSONArray sentences = json.getJSONArray("sentences");
		this.tokens = new String[sentences.size()][];
		this.posTags = new PoSTag[sentences.size()][];
		this.dependencyParses = new DependencyParse[sentences.size()];
		this.constituencyParses = new ConstituencyParse[sentences.size()];
		
		for (int i = 0; i < sentences.size(); i++) {
			JSONObject sentenceJson = sentences.getJSONObject(i);
			JSONArray tokensJson = sentenceJson.getJSONArray("tokens");
			JSONArray posTagsJson = (sentenceJson.has("posTags")) ? sentenceJson.getJSONArray("posTags") : null;
			
			this.tokens[i] = new String[tokensJson.size()];
			for (int j = 0; j < tokensJson.size(); j++)
				this.tokens[i][j] = tokensJson.getString(j);
			
			if (posTagsJson != null) {
				this.posTags[i] = new PoSTag[posTagsJson.size()];
				for (int j = 0; j < posTagsJson.size(); j++)
					this.posTags[i][j] = PoSTag.valueOf(posTagsJson.getString(j));
			}
			
			if (sentenceJson.has("dependencyParse"))
				this.dependencyParses[i] = DependencyParse.fromString(sentenceJson.getString("dependencyParse"), this, i);
			if (sentenceJson.has("constituencyParse"))
				this.constituencyParses[i] = ConstituencyParse.fromString(sentenceJson.getString("constituencyParse"), this, i);
		}
		
		return true;
	}

	@Override
	public Document makeInstanceFromJSONFile(String path) {
		return new DocumentInMemory(path);
	}
}
