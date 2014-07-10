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

package ark.model.annotator.nlp;

import ark.data.annotation.Document;
import ark.data.annotation.Language;
import ark.data.annotation.nlp.ConstituencyParse;
import ark.data.annotation.nlp.DependencyParse;
import ark.data.annotation.nlp.PoSTag;

/**
 * NLPAnnotator is an abstract parent for classes that 
 * supplement text with NLP 
 * annotations (parses, part-of-speech tags, etc) using
 * various pipelines.
 * 
 * The returned NLP annotations are in the ARKWater 
 * (https://github.com/forkunited/ARKWater)
 * library's format.
 * 
 * Once constructed for a specified language, an annotator 
 * object can be used by calling
 * the setText method to set the text to be annotated, and
 * then calling the make[X] methods to retrieve the annotations
 * for that text.
 * 
 * @author Bill McDowell
 *
 */
public abstract class NLPAnnotator {
	protected Language language; // language of the text
	protected String text; // text to annotate
	
	// Disable various components of the NLP pipeline
	protected boolean disabledPoSTags;
	protected boolean disabledDependencyParses;
	protected boolean disabledConstituencyParses;
	
	/**
	 * @param language
	 * @return true if the language of the annotator is successfully 
	 * set. 
	 */
	public abstract boolean setLanguage(Language language);
	
	/**
	 * @param text
	 * @return true if the annotator has received the text 
	 * and is ready to return annotations for it
	 */
	public abstract boolean setText(String text);
	
	/**
	 * @return a name for the annotator
	 */
	public abstract String toString();
	
	/**
	 * @return an array of tokens for each segmented 
	 * sentence of the text.
	 */
	public abstract String[][] makeTokens();
	
	/**
	 * @return an array of pos-tags for each segmented 
	 * sentence of the text.
	 */
	protected abstract PoSTag[][] makePoSTagsInternal();
	
	/**
	 * @return a dependency parse for each segmented 
	 * sentence of the text.
	 */
	protected abstract DependencyParse[] makeDependencyParsesInternal(Document document, int sentenceIndexOffset);
	
	/**
	 * @return a constituency parse for each segmented 
	 * sentence of the text.
	 */
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