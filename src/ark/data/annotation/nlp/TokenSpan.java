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

package ark.data.annotation.nlp;

import net.sf.json.JSONObject;
import ark.data.annotation.Document;

/**
 * TokenSpan represents a contiguous span of tokens in a document.  
 * 
 * @author Bill
 */
public class TokenSpan {
	private Document document;
	private int sentenceIndex;
	private int startTokenIndex; // 0-based token index (inclusive)
	private int endTokenIndex; // 0-based token index (exclusive)
	
	public TokenSpan(Document document, int sentenceIndex, int startTokenIndex, int endTokenIndex) {
		this.document = document;
		this.sentenceIndex = sentenceIndex;
		this.startTokenIndex = startTokenIndex;
		this.endTokenIndex = endTokenIndex;
	}
	
	public boolean containsToken(int sentenceIndex, int tokenIndex) {
		return this.sentenceIndex == sentenceIndex
				&& this.startTokenIndex <= tokenIndex
				&& this.endTokenIndex > tokenIndex;
	}
	
	public Document getDocument() {
		return this.document;
	}
	
	public int getSentenceIndex() {
		return this.sentenceIndex;
	}
	
	public int getStartTokenIndex() {
		return this.startTokenIndex;
	}
	
	public int getEndTokenIndex() {
		return this.endTokenIndex;
	}
	
	public String toString() {
		StringBuilder str = new StringBuilder();
		
		for (int i = this.startTokenIndex; i < this.endTokenIndex; i++)
			str.append(this.document.getToken(this.sentenceIndex, i)).append(" ");
		
		return str.toString().trim();
	}
	
	public JSONObject toJSON() {
		JSONObject json = new JSONObject();
		
		json.put("startTokenIndex", this.startTokenIndex);
		json.put("endTokenIndex", this.endTokenIndex);
		
		return json;
	}
	
	public static TokenSpan fromJSON(JSONObject json, Document document, int sentenceIndex) {
		return new TokenSpan(
			document,
			sentenceIndex,
			json.getInt("startTokenIndex"),
			json.getInt("endTokenIndex")
		);
	}
}

