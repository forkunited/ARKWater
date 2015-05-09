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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import edu.cmu.ml.rtw.annotation.DocumentAnnotation;
import ark.data.DataTools;
import ark.data.annotation.AnnotationType;
import ark.data.annotation.nlp.ConstituencyParse;
import ark.data.annotation.nlp.DependencyParse;
import ark.data.annotation.nlp.PoSTag;
import ark.data.annotation.nlp.Token;
import ark.data.annotation.nlp.TokenSpan;
import ark.data.annotation.nlp.TokenSpan.Relation;
import ark.model.annotator.nlp.PipelineNLP;
import ark.util.Pair;

/**
 * DocumentInMemory represents a text document with various 
 * NLP annotations (e.g. PoS tags, parses, etc) kept in 
 * memory.  
 * 
 * @author Bill McDowell
 * 
 */
public class DocumentNLPInMemory extends DocumentNLP {
	private String languageAnnotatorName;
	private String originalTextAnnotatorName;
	private String tokenAnnotatorName;
	private String posAnnotatorName;
	private String dependencyParseAnnotatorName;
	private String constituencyParseAnnotatorName;
	private String nerAnnotatorName;
	private String corefAnnotatorName;
	private Map<AnnotationTypeNLP<?>, String> otherAnnotatorNames;
	
	protected Language language;
	protected String originalText;
	protected Token[][] tokens;
	protected PoSTag[][] posTags;
	protected DependencyParse[] dependencyParses; 
	protected ConstituencyParse[] constituencyParses;
	protected Map<Integer, List<Pair<TokenSpan, String>>> ner;
	protected Map<Integer, List<Pair<TokenSpan, TokenSpanCluster>>> coref;
	
	private Map<AnnotationTypeNLP<?>, Object> otherDocumentAnnotations;
	private Map<AnnotationTypeNLP<?>, Map<Integer, Object>> otherSentenceAnnotations;
	private Map<AnnotationTypeNLP<?>, Map<Integer, List<Pair<TokenSpan, Object>>>> otherTokenSpanAnnotations;
	private Map<AnnotationTypeNLP<?>, Object[][]> otherTokenAnnotations;
	
	public DocumentNLPInMemory(DataTools dataTools) {
		super(dataTools);
	}

	public DocumentNLPInMemory(DataTools dataTools, JSONObject json) {
		super(dataTools, json);
	}
	
	public DocumentNLPInMemory(DataTools dataTools, DocumentAnnotation documentAnnotation) {
		super(dataTools, documentAnnotation);
	}
	
	public DocumentNLPInMemory(DataTools dataTools, String jsonPath) {
		super(dataTools, jsonPath);
	}
	
	public DocumentNLPInMemory(DataTools dataTools, String name, String text, Language language, PipelineNLP pipeline) {
		super(dataTools);
		
		this.name = name;
		this.language = language;
		
		this.originalText = text;
		if (!pipeline.setDocument(this))
			throw new IllegalArgumentException();
		
		if (pipeline.meetsAnnotatorRequirements(AnnotationTypeNLP.TOKEN, this)) {
			this.tokenAnnotatorName = pipeline.getAnnotatorName(AnnotationTypeNLP.TOKEN);
			this.tokens = pipeline.annotateTokens(AnnotationTypeNLP.TOKEN);
		}
		
		if (pipeline.meetsAnnotatorRequirements(AnnotationTypeNLP.POS, this)) {
			this.posAnnotatorName = pipeline.getAnnotatorName(AnnotationTypeNLP.POS);
			this.posTags = pipeline.annotateTokens(AnnotationTypeNLP.POS);
		}
		
		if (pipeline.meetsAnnotatorRequirements(AnnotationTypeNLP.CONSTITUENCY_PARSE, this)) {
			this.constituencyParseAnnotatorName = pipeline.getAnnotatorName(AnnotationTypeNLP.CONSTITUENCY_PARSE);
			Map<Integer, ConstituencyParse> parses = pipeline.annotateSentences(AnnotationTypeNLP.CONSTITUENCY_PARSE);
			this.constituencyParses = new ConstituencyParse[this.tokens.length];
			for (Entry<Integer, ConstituencyParse> parse : parses.entrySet())
				this.constituencyParses[parse.getKey()] = parse.getValue();
		}
		
		if (pipeline.meetsAnnotatorRequirements(AnnotationTypeNLP.DEPENDENCY_PARSE, this)) {
			this.dependencyParseAnnotatorName = pipeline.getAnnotatorName(AnnotationTypeNLP.DEPENDENCY_PARSE);
			Map<Integer, DependencyParse> parses = pipeline.annotateSentences(AnnotationTypeNLP.DEPENDENCY_PARSE);
			this.dependencyParses = new DependencyParse[this.tokens.length];
			for (Entry<Integer, DependencyParse> parse : parses.entrySet())
				this.dependencyParses[parse.getKey()] = parse.getValue();
		}
		
		if (pipeline.meetsAnnotatorRequirements(AnnotationTypeNLP.NER, this)) {
			this.nerAnnotatorName = pipeline.getAnnotatorName(AnnotationTypeNLP.NER);
			List<Pair<TokenSpan, String>> ner = pipeline.annotateTokenSpans(AnnotationTypeNLP.NER);
			this.ner = new HashMap<Integer, List<Pair<TokenSpan, String>>>();
			for (Pair<TokenSpan, String> nerSpan : ner) {
				if (!this.ner.containsKey(nerSpan.getFirst().getSentenceIndex()))
					this.ner.put(nerSpan.getFirst().getSentenceIndex(), new ArrayList<Pair<TokenSpan, String>>());
				this.ner.get(nerSpan.getFirst().getSentenceIndex()).add(nerSpan);
			}
		}
		
		if (pipeline.meetsAnnotatorRequirements(AnnotationTypeNLP.COREF, this)) {
			this.corefAnnotatorName = pipeline.getAnnotatorName(AnnotationTypeNLP.COREF);
			List<Pair<TokenSpan, TokenSpanCluster>> coref = pipeline.annotateTokenSpans(AnnotationTypeNLP.COREF);
			this.coref = new HashMap<Integer, List<Pair<TokenSpan, TokenSpanCluster>>>();
			for (Pair<TokenSpan, TokenSpanCluster> corefSpan : coref) {
				if (!this.coref.containsKey(corefSpan.getFirst().getSentenceIndex()))
					this.coref.put(corefSpan.getFirst().getSentenceIndex(), new ArrayList<Pair<TokenSpan, TokenSpanCluster>>());
				this.coref.get(corefSpan.getFirst().getSentenceIndex()).add(corefSpan);
			}
		}
		
		this.originalText = null;
	}
	
	@Override
	public JSONObject toJSON() {
		JSONObject json = new JSONObject();
		JSONArray sentencesJson = new JSONArray();
		
		try {
			JSONObject annotators = new JSONObject();
			annotators.put(AnnotationTypeNLP.ORIGINAL_TEXT.getType(), this.originalTextAnnotatorName);
			annotators.put(AnnotationTypeNLP.LANGUAGE.getType(), this.languageAnnotatorName);
			annotators.put(AnnotationTypeNLP.TOKEN.getType(), this.tokenAnnotatorName);
			annotators.put(AnnotationTypeNLP.POS.getType(), this.posAnnotatorName);
			annotators.put(AnnotationTypeNLP.CONSTITUENCY_PARSE.getType(), this.constituencyParseAnnotatorName);
			annotators.put(AnnotationTypeNLP.DEPENDENCY_PARSE.getType(), this.dependencyParseAnnotatorName);
			annotators.put(AnnotationTypeNLP.NER.getType(), this.nerAnnotatorName);
			annotators.put(AnnotationTypeNLP.COREF.getType(), this.corefAnnotatorName);
			
			if (this.otherAnnotatorNames != null) {
				for (Entry<AnnotationTypeNLP<?>, String> entry : this.otherAnnotatorNames.entrySet()) {
					annotators.put(entry.getKey().getType(), entry.getValue());
				}
			}
			
			if (annotators.length() > 0)
				json.put("annotators", annotators);
			
			json.put("name", this.name);
			json.put("text", this.originalText);
			
			if (this.language != null)
				json.put("language", this.language.toString());
			
			if (this.otherDocumentAnnotations != null) {
				for (Entry<AnnotationTypeNLP<?>, Object> entry : this.otherDocumentAnnotations.entrySet()) 
					json.put(entry.getKey().getType(), entry.getKey().serialize(entry.getValue()));
			}
			
			int sentenceCount = getSentenceCount();
			for (int i = 0; i < sentenceCount; i++) {
				int tokenCount = getSentenceTokenCount(i);
				JSONObject sentenceJson = new JSONObject();
				sentenceJson.put("sentence", getSentence(i));
				
				JSONArray tokensJson = new JSONArray();
				JSONArray posTagsJson = new JSONArray();
				
				for (int j = 0; j < tokenCount; j++) {
					Token token = this.tokens[i][j];
					if (token.getCharSpanEnd() < 0 || token.getCharSpanStart() < 0)
						tokensJson.put(getToken(i, j).getStr());
					else
						tokensJson.put(token.toJSON());
					
					if (this.posTags != null) {
						PoSTag posTag = getPoSTag(i, j);
						if (posTag != null)
							posTagsJson.put(posTag.toString());	
					}
				}
				
				sentenceJson.put("tokens", tokensJson);
				if (this.posTags != null)
					sentenceJson.put("posTags", posTagsJson);
				if (this.dependencyParses != null)
					sentenceJson.put("dependencyParse", getDependencyParse(i).toString());
				if (this.constituencyParses != null && getConstituencyParse(i) != null)
					sentenceJson.put("constituencyParse", getConstituencyParse(i).toString());
				
				if (this.otherSentenceAnnotations != null) {
					for (Entry<AnnotationTypeNLP<?>, Map<Integer, Object>> entry : this.otherSentenceAnnotations.entrySet()) {
						if (entry.getValue().containsKey(i))
							sentenceJson.put(entry.getKey().getType(), entry.getKey().serialize(entry.getValue().get(i)));
					}
				}
				
				if (this.otherTokenAnnotations != null) {
					for (Entry<AnnotationTypeNLP<?>, Object[][]> entry : this.otherTokenAnnotations.entrySet()) {
						JSONArray jsonObjs = new JSONArray();
						
						for (int j = 0; j < entry.getValue().length; j++)
							jsonObjs.put(entry.getKey().serialize(entry.getValue()[i][j]));
						
						sentenceJson.put(entry.getKey().getType() + "s", jsonObjs);
					}
				}
				
				sentencesJson.put(sentenceJson);
			}
			json.put("sentences", sentencesJson);

			if (this.ner != null) {
				JSONArray nerJson = new JSONArray();
				for (Entry<Integer, List<Pair<TokenSpan, String>>> sentenceEntry : this.ner.entrySet()) {
					JSONObject sentenceJson = new JSONObject();
					sentenceJson.put("sentence", sentenceEntry.getKey());
					JSONArray annotationSpansJson = new JSONArray();
					for (Pair<TokenSpan, String> annotationSpan : sentenceEntry.getValue()) {
						JSONObject annotationSpanJson = new JSONObject();
						
						annotationSpanJson.put("tokenSpan", annotationSpan.getFirst().toJSON(false));
						annotationSpanJson.put("type", AnnotationTypeNLP.NER.serialize(annotationSpan.getSecond()));
						annotationSpansJson.put(annotationSpanJson);
					}
					
					sentenceJson.put("nerSpans", annotationSpansJson);
					nerJson.put(sentenceJson);
				}
				json.put(AnnotationTypeNLP.NER.getType(), nerJson);
			}
			
			if (this.coref != null) {
				JSONArray corefJson = new JSONArray();
				for (Entry<Integer, List<Pair<TokenSpan, TokenSpanCluster>>> sentenceEntry : this.coref.entrySet()) {
					JSONObject sentenceJson = new JSONObject();
					sentenceJson.put("sentence", sentenceEntry.getKey());
					JSONArray annotationSpansJson = new JSONArray();
					for (Pair<TokenSpan, TokenSpanCluster> annotationSpan : sentenceEntry.getValue()) {
						JSONObject annotationSpanJson = new JSONObject();
						
						annotationSpanJson.put("tokenSpan", annotationSpan.getFirst().toJSON(false));
						annotationSpanJson.put("type", AnnotationTypeNLP.COREF.serialize(annotationSpan.getSecond()));
						annotationSpansJson.put(annotationSpanJson);
					}
					
					sentenceJson.put("corefSpans", annotationSpansJson);
					corefJson.put(sentenceJson);
				}
				json.put(AnnotationTypeNLP.COREF.getType(), corefJson);
			}
			
			if (this.otherTokenSpanAnnotations != null) {
				for (Entry<AnnotationTypeNLP<?>, Map<Integer, List<Pair<TokenSpan, Object>>>> entry : this.otherTokenSpanAnnotations.entrySet()) {
					JSONArray annotationsJson = new JSONArray();
					String spansStr = entry.getKey().toString() + "Spans";
					for (Entry<Integer, List<Pair<TokenSpan, Object>>> sentenceEntry : entry.getValue().entrySet()) {
						JSONObject sentenceJson = new JSONObject();
						sentenceJson.put("sentence", sentenceEntry.getKey());
						JSONArray annotationSpansJson = new JSONArray();
						for (Pair<TokenSpan, Object> annotationSpan : sentenceEntry.getValue()) {
							JSONObject annotationSpanJson = new JSONObject();
							
							annotationSpanJson.put("tokenSpan", annotationSpan.getFirst().toJSON(false));
							annotationSpanJson.put("type", entry.getKey().serialize(annotationSpan.getSecond()));
							annotationSpansJson.put(annotationSpanJson);
						}
						
						sentenceJson.put(spansStr, annotationSpansJson);
						annotationsJson.put(sentenceJson);
					}
					
					json.put(entry.getKey().getType(), annotationsJson);
				}
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
		return json;
	}
	
	@Override
	public boolean fromJSON(JSONObject json) {
		throw new UnsupportedOperationException();
		/* FIXME Finish this try {
			if (json.has("annotators")) {
				JSONObject annotatorsJson = json.getJSONObject("annotators");
				String[] annotatorTypes = JSONObject.getNames(annotatorsJson);
				for (String annotatorType : annotatorTypes) {
					if (annotatorType.equals(AnnotationTypeNLP.ORIGINAL_TEXT))
						this.originalTextAnnotatorName = annotatorsJson.getString(annotatorType);
					else if (annotatorType.equals(AnnotationTypeNLP.LANGUAGE))
						this.languageAnnotatorName = annotatorsJson.getString(annotatorType);
					else if (annotatorType.equals(AnnotationTypeNLP.TOKEN))
						this.tokenAnnotatorName = annotatorsJson.getString(annotatorType);
					else if (annotatorType.equals(AnnotationTypeNLP.POS))
						this.posAnnotatorName = annotatorsJson.getString(annotatorType);
					else if (annotatorType.equals(AnnotationTypeNLP.CONSTITUENCY_PARSE))
						this.constituencyParseAnnotatorName = annotatorsJson.getString(annotatorType);
					else if (annotatorType.equals(AnnotationTypeNLP.DEPENDENCY_PARSE))
						this.dependencyParseAnnotatorName = annotatorsJson.getString(annotatorType);
					else if (annotatorType.equals(AnnotationTypeNLP.NER))
						this.nerAnnotatorName = annotatorsJson.getString(annotatorType);
					else if (annotatorType.equals(AnnotationTypeNLP.COREF))
						this.corefAnnotatorName = annotatorsJson.getString(annotatorType);
					else {
						if (this.otherAnnotatorNames == null)
							this.otherAnnotatorNames = new HashMap<AnnotationTypeNLP<?>, String>();
						this.otherAnnotatorNames.put(this.dataTools.getAnnotationTypeNLP(annotatorType), annotatorsJson.getString(annotatorType));
					}
				}
			}
			
			if (json.has("name"))
				this.name = json.getString("name");
			
			if (json.has("text"))
				this.originalText = json.getString("text");
			
			if (json.has("language"))
				this.language = Language.valueOf(json.getString("language"));
			
			// FIXME Do other document
			
			JSONArray sentences = json.getJSONArray("sentences");
			this.tokens = new Token[sentences.length()][];
			this.posTags = new PoSTag[sentences.length()][];
			this.dependencyParses = new DependencyParse[sentences.length()];
			this.constituencyParses = new ConstituencyParse[sentences.length()];
			
			int characterOffset = 0;
			for (int i = 0; i < sentences.length(); i++) {
				JSONObject sentenceJson = sentences.getJSONObject(i);
				JSONArray tokensJson = sentenceJson.getJSONArray("tokens");
				JSONArray posTagsJson = (sentenceJson.has("posTags")) ? sentenceJson.getJSONArray("posTags") : null;
				
				this.tokens[i] = new Token[tokensJson.length()];
				for (int j = 0; j < tokensJson.length(); j++) {
					JSONObject tokenJson = tokensJson.optJSONObject(j);
					if (tokenJson == null) {
						String tokenStr = tokensJson.getString(j);
						this.tokens[i][j] = new Token(this, tokenStr, characterOffset, characterOffset + tokenStr.length());
						characterOffset += tokenStr.length() + 1;
					} else {
						Token token = new Token(this);
						if (!token.fromJSON(tokenJson))
							return false;
						this.tokens[i][j] = token;
					}
				}
				
				if (posTagsJson != null) {
					this.posTags[i] = new PoSTag[posTagsJson.length()];
					for (int j = 0; j < posTagsJson.length(); j++)
						this.posTags[i][j] = PoSTag.valueOf(posTagsJson.getString(j));
				}
				
				if (sentenceJson.has("dependencyParse"))
					this.dependencyParses[i] = DependencyParse.fromString(sentenceJson.getString("dependencyParse"), this, i);
				if (sentenceJson.has("constituencyParse"))
					this.constituencyParses[i] = ConstituencyParse.fromString(sentenceJson.getString("constituencyParse"), this, i);
			
				// FIXME Do other sentence
				// FIXME Do other tokens
			}
		
			// FIXME Do ner
			// FIXME Do coref
			
			// FIXME Do other tokenspans
			if (this.dataTools != null) {
				this.otherTokenSpanAnnotations = new HashMap<AnnotationTypeTokenSpan, Map<Integer, List<Pair<TokenSpan, Object>>>>();
				String[] jsonKeys = JSONObject.getNames(json);
				for (String jsonKey : jsonKeys) {
					AnnotationTypeTokenSpan annotationType = this.dataTools.getTokenSpanAnnotationType(jsonKey);
					if (annotationType == null 
							|| annotationType.equals(AnnotationTypeTokenSpan.TOKEN)
							|| annotationType.equals(AnnotationTypeTokenSpan.POS))
						continue;
					
					Map<Integer, List<Pair<TokenSpan, Object>>> sentenceAnnotations = new HashMap<Integer, List<Pair<TokenSpan, Object>>>();
					JSONArray annotations = json.getJSONArray(jsonKey);
					String spansStr = jsonKey + "Spans";
					for (int i = 0; i < annotations.length(); i++) {
						JSONObject annotation = annotations.getJSONObject(i);
						int sentenceIndex = annotation.getInt("sentence");
						JSONArray annotationSpansJson = annotation.getJSONArray(spansStr);
						List<Pair<TokenSpan, Object>> annotationSpans = new ArrayList<Pair<TokenSpan, Object>>();
						for (int j = 0; j < annotationSpansJson.length(); j++) {
							JSONObject annotationSpanJson = annotationSpansJson.getJSONObject(j);
							TokenSpan span = TokenSpan.fromJSON(annotationSpanJson.getJSONObject("tokenSpan"), this, sentenceIndex);
							
							Object type = null;
							if (annotationType.isSerializable()) {
								type = ((JSONSerializableAnnotation)annotationType.makeAnnotationInstance(this)).fromJSON(annotationSpanJson.getJSONObject("type"));
							} else {
								type = annotationSpanJson.getString("type");
							}
							
							annotationSpans.add(new Pair<TokenSpan, Object>(span, type));
						}
						
						sentenceAnnotations.put(sentenceIndex, annotationSpans);
					}
					
					this.otherTokenSpanAnnotations.put(annotationType, sentenceAnnotations);
				}
			}
		} catch (JSONException e) {
			e.printStackTrace();
		}
		
		return true;*/
	}

	@Override
	public DocumentNLP makeInstanceFromJSONFile(String path) {
		return new DocumentNLPInMemory(this.dataTools, path);
	}
	
	@Override
	public DocumentNLP makeInstanceFromRTWDocumentAnnotation(DocumentAnnotation documentAnnotation, Map<AnnotationTypeNLP<?>, String> annotators) {
		return new DocumentNLPInMemory(this.dataTools, documentAnnotation);
	}

	@Override
	public boolean fromRTWDocumentAnnotation(DocumentAnnotation documentAnnotation, Map<AnnotationTypeNLP<?>, String> annotators) {
		throw new UnsupportedOperationException();
		/* FIXME finish this.name = documentAnnotation.getDocumentId();
		this.nlpAnnotator = ""; // FIXME: Assumes same annotator for everything
		this.language = Language.Unknown;
		
		List<Annotation> annotations = documentAnnotation.getAllAnnotations();
		TreeMap<Integer, Token> tokenMap = new TreeMap<Integer, Token>();
		TreeMap<Integer, PoSTag> posMap = new TreeMap<Integer, PoSTag>();
		TreeMap<Integer, Integer> sentenceSpanMap = new TreeMap<Integer, Integer>();
		TreeMap<Integer, String> dependencyMap = new TreeMap<Integer, String>();
		TreeMap<Integer, String> constituencyMap = new TreeMap<Integer, String>();
		Map<AnnotationTypeTokenSpan, TreeMap<Integer, Pair<Integer, Object>>> tokenSpanAnnotationsMap = new HashMap<AnnotationTypeTokenSpan, TreeMap<Integer, Pair<Integer, Object>>>();
		
		for (Annotation annotation : annotations) {
			if (annotation.getSlot().equals("language")) {
				this.language = Language.valueOf(annotation.getStringValue());
			} else if (annotation.getSlot().equals("token")) {
				tokenMap.put(annotation.getSpanStart(), new Token(annotation.getStringValue(), annotation.getSpanStart(), annotation.getSpanEnd()));
			} else if (annotation.getSlot().equals("pos")) {
				posMap.put(annotation.getSpanStart(), PoSTag.valueOf(annotation.getStringValue()));
			} else if (annotation.getSlot().equals("sentence")) {
				sentenceSpanMap.put(annotation.getSpanStart(), annotation.getSpanEnd());
			} else if (annotation.getSlot().equals("dependency_parse")) {
				dependencyMap.put(annotation.getSpanStart(), annotation.getStringValue());
			} else if (annotation.getSlot().equals("constituency_parse")) {
				constituencyMap.put(annotation.getSpanStart(), annotation.getStringValue());
			} else if (this.dataTools.getTokenSpanAnnotationType(annotation.getSlot()) != null) {
				AnnotationTypeTokenSpan annotationType = this.dataTools.getTokenSpanAnnotationType(annotation.getSlot());
				Object annotationObj = null;
				if (annotationType.isSerializable())
					annotationObj = annotation.getJsonValue();
				else
					annotationObj = annotation.getStringValue();
				
				if (!tokenSpanAnnotationsMap.containsKey(annotationType))
					tokenSpanAnnotationsMap.put(annotationType, new TreeMap<Integer, Pair<Integer, Object>>());
				tokenSpanAnnotationsMap.get(annotationType).put(annotation.getSpanStart(), new Pair<Integer, Object>(annotation.getSpanEnd(), annotationObj));
			}
		}

		this.tokens = new Token[sentenceSpanMap.size()][];
		if (posMap.size() > 0)
			this.posTags = new PoSTag[sentenceSpanMap.size()][];
		else 
			this.posTags = new PoSTag[0][];
		
		if (dependencyMap.size() > 0)
			this.dependencyParses = new DependencyParse[dependencyMap.size()];
		else
			this.dependencyParses = new DependencyParse[0];
		
		if (constituencyMap.size() > 0)
			this.constituencyParses = new ConstituencyParse[constituencyMap.size()];
		else
			this.constituencyParses = new ConstituencyParse[0];
		
		this.otherTokenSpanAnnotations = new HashMap<AnnotationTypeTokenSpan, Map<Integer, List<Pair<TokenSpan, Object>>>>();
		for (AnnotationTypeTokenSpan tokenSpanAnnotationType : tokenSpanAnnotationsMap.keySet())
			this.otherTokenSpanAnnotations.put(tokenSpanAnnotationType, new HashMap<Integer, List<Pair<TokenSpan, Object>>>());
		
		int sentenceIndex = 0;
		for (Entry<Integer, Integer> entry : sentenceSpanMap.entrySet()) {
			int sentenceStart = entry.getKey();
			int sentenceEnd = entry.getValue();
			
			SortedMap<Integer, Token> sentenceTokens = tokenMap.subMap(sentenceStart, sentenceEnd);
			this.tokens[sentenceIndex] = new Token[sentenceTokens.size()];
			int tokenIndex = 0;
			for (Token token : sentenceTokens.values()) {
				this.tokens[sentenceIndex][tokenIndex] = token;
				tokenIndex++;
			}
			
			if (posMap.size() > 0) {
				SortedMap<Integer, PoSTag> sentencePoS = posMap.subMap(sentenceStart, sentenceEnd);
				if (sentencePoS.size() != sentenceTokens.size())
					return false;
				int posIndex = 0;
				for (PoSTag pos : sentencePoS.values()) {
					this.posTags[sentenceIndex][posIndex] = pos;
					posIndex++;
				}
			}
			
			if (dependencyMap.size() > 0) {
				if (!dependencyMap.containsKey(sentenceStart))
					return false;
				this.dependencyParses[sentenceIndex] = DependencyParse.fromString(dependencyMap.get(sentenceStart), this, sentenceIndex);
			}
			
			if (constituencyMap.size() > 0) {
				if (!constituencyMap.containsKey(sentenceStart))
					return false;
				this.constituencyParses[sentenceIndex] = ConstituencyParse.fromString(constituencyMap.get(sentenceStart), this, sentenceIndex);
			}
		
			
			for (Entry<AnnotationTypeTokenSpan, TreeMap<Integer, Pair<Integer, Object>>> tokenSpanAnnotationEntry : tokenSpanAnnotationsMap.entrySet()) {
				SortedMap<Integer, Pair<Integer, Object>> sentenceTokenSpanAnnotations = tokenSpanAnnotationEntry.getValue().subMap(sentenceStart, sentenceEnd);
				if (sentenceTokenSpanAnnotations.size() == 0)
					continue;
				
				Map<Integer, List<Pair<TokenSpan, Object>>> otherAnnotations = this.otherTokenSpanAnnotations.get(tokenSpanAnnotationEntry.getKey());
				List<Pair<TokenSpan, Object>> sentenceAnnotations = new ArrayList<Pair<TokenSpan, Object>>();
				for (Entry<Integer, Pair<Integer, Object>> sentenceTokenSpanAnnotationEntry : sentenceTokenSpanAnnotations.entrySet()) {
					int charStart = sentenceTokenSpanAnnotationEntry.getKey();
					int charEnd = sentenceTokenSpanAnnotationEntry.getValue().getFirst();
					int startTokenIndex = -1;
					int endTokenIndex = -1;
					for (int i = 0; i < this.tokens[sentenceIndex].length; i++) {
						if (this.tokens[sentenceIndex][i].getCharSpanStart() == charStart)
							startTokenIndex = i;
						if (this.tokens[sentenceIndex][i].getCharSpanEnd() == charEnd) {
							endTokenIndex = i + 1;
							break;
						}
					}
					
					if (startTokenIndex < 0 || endTokenIndex < 0)
						return false;
					
					TokenSpan span = new TokenSpan(this, sentenceIndex, startTokenIndex, endTokenIndex);
					Object annotationObj = sentenceTokenSpanAnnotationEntry.getValue().getSecond();
					sentenceAnnotations.add(new Pair<TokenSpan, Object>(span, annotationObj));
				}
				
				otherAnnotations.put(sentenceIndex, sentenceAnnotations);
			}
			
			sentenceIndex++;
		}
		
		return true;*/
	}
	
	@Override
	public DocumentAnnotation toRTWDocumentAnnotation(Collection<AnnotationTypeNLP<?>> annotationTypes) {
		throw new UnsupportedOperationException();
		/* FIXME Finish this later
		DateTime annotationTime = DateTime.now();
		List<Annotation> annotations = new ArrayList<Annotation>();
		annotations.add(new Annotation(-1, 
									   -1, 
									   "language", 
									   this.languageAnnotatorName != null ? this.languageAnnotatorName : "", 
									   this.name, 
									   this.language.toString(), 
									   null, 
									   1.0,
									   annotationTime, 
									   null));
		
		for (int i = 0; i < this.tokens.length; i++) {
			for (int j = 0; j < this.tokens[i].length; j++) {
				annotations.add(new Annotation(this.tokens[i][j].getCharSpanStart(), 
											   this.tokens[i][j].getCharSpanEnd(), 
											   "token", 
											   this.nlpAnnotator, 
											   this.name, 
											   this.tokens[i][j].getStr(), 
											   null, 
											   1.0,
											   annotationTime, 
						   					   ""));
				
				annotations.add(new Annotation(this.tokens[i][j].getCharSpanStart(), 
						   this.tokens[i][j].getCharSpanEnd(), 
						   "pos", 
						   this.nlpAnnotator, 
						   this.name, 
						   this.posTags[i][j].toString(), 
						   null, 
						   1.0,
						   annotationTime, 
	   					   ""));
			}
			
			annotations.add(new Annotation(this.tokens[i][0].getCharSpanStart(), 
					   this.tokens[i][this.tokens[i].length - 1].getCharSpanEnd(), 
					   "sentence", 
					   this.nlpAnnotator, 
					   this.name, 
					   getSentence(i), 
					   null, 
					   1.0,
					   annotationTime, 
					   ""));
			
			if (this.dependencyParses.length > 0) {
				annotations.add(new Annotation(this.tokens[i][0].getCharSpanStart(), 
						   this.tokens[i][this.tokens[i].length - 1].getCharSpanEnd(), 
						   "dependency_parse", 
						   this.nlpAnnotator, 
						   this.name, 
						   getDependencyParse(i).toString(), 
						   null, 
						   1.0,
						   annotationTime, 
						   ""));
			}
			
			if (this.constituencyParses.length > 0) {
				annotations.add(new Annotation(this.tokens[i][0].getCharSpanStart(), 
						   this.tokens[i][this.tokens[i].length - 1].getCharSpanEnd(), 
						   "constituency_parse", 
						   this.nlpAnnotator, 
						   this.name, 
						   getConstituencyParse(i).toString(), 
						   null, 
						   1.0,
						   annotationTime, 
						   ""));
			}
		}
		
		for (Entry<AnnotationTypeTokenSpan, Map<Integer, List<Pair<TokenSpan, Object>>>> entry : this.otherTokenSpanAnnotations.entrySet()) {
			for (Entry<Integer, List<Pair<TokenSpan, Object>>> sentenceEntry : entry.getValue().entrySet()) {
				for (Pair<TokenSpan, Object> annotation : sentenceEntry.getValue()) {
					TokenSpan span = annotation.getFirst();
					int charSpanStart = this.tokens[span.getSentenceIndex()][span.getStartTokenIndex()].getCharSpanStart();
					int charSpanEnd = this.tokens[span.getSentenceIndex()][span.getEndTokenIndex() - 1].getCharSpanEnd();
					
					annotations.add(new Annotation(charSpanStart, 
							   					   charSpanEnd, 
							   					   entry.getKey().toString(), 
							   					   this.nlpAnnotator, 
							   					   this.name, 
							   					   (entry.getKey().isSerializable()) ? null : annotation.getSecond().toString(), 
							   					   (entry.getKey().isSerializable()) ? ((JSONSerializableAnnotation)annotation.getSecond()).toJSON() : null, 
							   					   1.0,
							   					   annotationTime, 
							   					   ""));
				}
			}
		}
		
		return new DocumentAnnotation(this.name, annotations);*/
	}

	@Override
	public int getSentenceCount() {
		return this.tokens.length;
	}
	
	@Override
	public int getSentenceTokenCount(int sentenceIndex) {
		return this.tokens[sentenceIndex].length;
	}
	
	@Override
	public String getOriginalText() {
		return this.originalText;
	}
	
	@Override
	public String getText() {
		StringBuilder text = new StringBuilder();
		for (int i = 0; i < getSentenceCount(); i++)
			text = text.append(getSentence(i)).append(" ");
		return text.toString().trim();
	}
	
	@Override
	public String getSentence(int sentenceIndex) {
		StringBuilder sentenceStr = new StringBuilder();
		
		for (int i = 0; i < this.tokens[sentenceIndex].length; i++) {
			sentenceStr = sentenceStr.append(this.tokens[sentenceIndex][i]).append(" ");
		}
		return sentenceStr.toString().trim();
	}
	
	@Override
	public Token getToken(int sentenceIndex, int tokenIndex) {
		if (tokenIndex < 0)
			return new Token(this, "ROOT");
		else 
			return this.tokens[sentenceIndex][tokenIndex];
	}
	
	@Override
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
	
	@Override
	public Language getLanguage() {
		return this.language;
	}

	@Override
	public List<Pair<TokenSpan, String>> getNer(TokenSpan tokenSpan,
			Relation[] relationToAnnotations) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<Pair<TokenSpan, TokenSpanCluster>> getCoref(
			TokenSpan tokenSpan, Relation[] relationToAnnotations) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getAnnotatorName(AnnotationType<?> annotationType) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean hasAnnotationType(AnnotationType<?> annotationType) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public Collection<AnnotationType<?>> getAnnotationTypes() {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public <T> T getDocumentAnnotation(AnnotationTypeNLP<T> annotationType) {		
		T anno = super.getDocumentAnnotation(annotationType);
		if (anno != null)
			return anno;
		return annotationType.getAnnotationClass().cast(this.otherDocumentAnnotations.get(annotationType));
	}
	
	@Override
	public <T> T getSentenceAnnotation(AnnotationTypeNLP<T> annotationType, int sentenceIndex) {
		T anno = super.getSentenceAnnotation(annotationType, sentenceIndex);
		if (anno != null)
			return anno;
		
		Map<Integer, ?> sentenceAnnotation = this.otherSentenceAnnotations.get(annotationType);	
		return annotationType.getAnnotationClass().cast(sentenceAnnotation.get(sentenceIndex));
	}
	
	public <T> List<Pair<TokenSpan, T>> getTokenSpanAnnotations(AnnotationTypeNLP<T> annotationType, TokenSpan tokenSpan, TokenSpan.Relation[] relationsToAnnotations) {
		List<Pair<TokenSpan, T>> anno = super.getTokenSpanAnnotations(annotationType, tokenSpan, relationsToAnnotations);
		if (anno != null)
			return anno;
		List<Pair<TokenSpan, Object>> tokenSpanAnnotation = this.otherTokenSpanAnnotations.get(annotationType).get(tokenSpan.getSentenceIndex());
		if (tokenSpanAnnotation == null)
			return null;
		anno = new ArrayList<Pair<TokenSpan, T>>();
		for (Pair<TokenSpan, Object> span : tokenSpanAnnotation)
			anno.add(new Pair<TokenSpan, T>(span.getFirst(), annotationType.getAnnotationClass().cast(span.getSecond())));
		return anno;
	}
	
	public <T> T getTokenAnnotation(AnnotationTypeNLP<T> annotationType, int sentenceIndex, int tokenIndex) {	
		T anno = super.getTokenAnnotation(annotationType, sentenceIndex, tokenIndex);
		if (anno != null)
			return anno;
		
		return annotationType.getAnnotationClass().cast(this.otherTokenAnnotations.get(annotationType)[sentenceIndex][tokenIndex]);
	}
}
