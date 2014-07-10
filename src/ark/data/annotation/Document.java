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

/**
 * 
 * Document represents a JSON-serializable text document with 
 * various NLP annotations (e.g. PoS tags, parses, etc).  The methods
 * for getting the NLP annotations are kept abstract so 
 * that they can be implemented in ways that allow for
 * caching in cases when all of the documents don't fit
 * in memory.  In-memory implementations of these methods
 * are given by the ark.data.annotation.DocumentInMemory 
 * class.
 * 
 * @author Bill McDowell
 *
 */
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
