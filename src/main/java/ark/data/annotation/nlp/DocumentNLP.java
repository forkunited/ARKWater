package ark.data.annotation.nlp;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import org.json.JSONException;
import org.json.JSONObject;

import ark.data.DataTools;
import ark.data.annotation.Document;
import ark.util.FileUtil;
import ark.util.Pair;
import edu.cmu.ml.rtw.annotation.DocumentAnnotation;

public abstract class DocumentNLP extends Document {
	public DocumentNLP(DataTools dataTools) {
		super(dataTools);
	}
	
	public DocumentNLP(DataTools dataTools, JSONObject json) {
		this(dataTools);
		fromJSON(json);
	}
	
	public DocumentNLP(DataTools dataTools, DocumentAnnotation documentAnnotation) {
		this(dataTools);
		fromRTWDocumentAnnotation(documentAnnotation);
	}
	
	public DocumentNLP(DataTools dataTools, String jsonPath) {
		this(dataTools);
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
		
		try {
			if (!fromJSON(new JSONObject(lines.toString())))
				throw new IllegalArgumentException();
		} catch (JSONException e) {
			throw new IllegalArgumentException();
		}
	}
	
	public List<String> getSentenceTokenStrs(int sentenceIndex) {
		int sentenceTokenCount = getSentenceTokenCount(sentenceIndex);
		List<String> sentenceTokens = new ArrayList<String>(sentenceTokenCount);
		for (int i = 0; i < sentenceTokenCount; i++)
			sentenceTokens.add(getTokenStr(sentenceIndex, i));
		return sentenceTokens;
	}
	
	public List<PoSTag> getSentencePoSTags(int sentenceIndex) {
		int sentenceTokenCount = getSentenceTokenCount(sentenceIndex);
		List<PoSTag> sentencePoSTags = new ArrayList<PoSTag>(sentenceTokenCount);
		for (int i = 0; i < sentenceTokenCount; i++)
			sentencePoSTags.add(getPoSTag(sentenceIndex, i));
		return sentencePoSTags;
	}
	
	public String getTokenStr(int sentenceIndex, int tokenIndex) {
		return getToken(sentenceIndex, tokenIndex).getStr();
	}
	

	public List<Pair<TokenSpan, String>> getNer(TokenSpan tokenSpan) {
		return getNer(tokenSpan, TokenSpan.ANY_RELATION);
	}
	
	public List<Pair<TokenSpan, TokenSpanCluster>> getCoref(TokenSpan tokenSpan) {
		return getCoref(tokenSpan, TokenSpan.ANY_RELATION);
	}
	
	public <T> T getDocumentAnnotation(AnnotationTypeNLP<T> annotationType) {
		if (annotationType.equals(AnnotationTypeNLP.LANGUAGE))
			return annotationType.getAnnotationClass().cast(getLanguage());
		else if (annotationType.equals(AnnotationTypeNLP.ORIGINAL_TEXT))
			return annotationType.getAnnotationClass().cast(getOriginalText());
		else
			return null;
	}
	
	public <T> T getSentenceAnnotation(AnnotationTypeNLP<T> annotationType, int sentenceIndex) {
		if (annotationType.equals(AnnotationTypeNLP.CONSTITUENCY_PARSE)) {
			return annotationType.getAnnotationClass().cast(getConstituencyParse(sentenceIndex));
		} else if (annotationType.equals(AnnotationTypeNLP.DEPENDENCY_PARSE)) {
			return annotationType.getAnnotationClass().cast(getDependencyParse(sentenceIndex));
		} else {
			return null;
		}
	}

	public <T> List<Pair<TokenSpan, T>> getTokenSpanAnnotations(AnnotationTypeNLP<T> annotationType, TokenSpan tokenSpan) {
		return getTokenSpanAnnotations(annotationType, tokenSpan, TokenSpan.ANY_RELATION);
	}
	
	public <T> List<Pair<TokenSpan, T>> getTokenSpanAnnotations(AnnotationTypeNLP<T> annotationType, TokenSpan tokenSpan, TokenSpan.Relation[] relationsToAnnotations) {
		if (annotationType.equals(AnnotationTypeNLP.NER)) {
			List<Pair<TokenSpan, String>> ner = getNer(tokenSpan, relationsToAnnotations);
			List<Pair<TokenSpan, T>> cast = new ArrayList<Pair<TokenSpan, T>>(ner.size() * 2);
			for (Pair<TokenSpan, String> nerSpan : ner)
				cast.add(new Pair<TokenSpan, T>(nerSpan.getFirst(), annotationType.getAnnotationClass().cast(nerSpan.getSecond())));
			return cast;
		} else if (annotationType.equals(AnnotationTypeNLP.COREF)) {
			List<Pair<TokenSpan, TokenSpanCluster>> coref = getCoref(tokenSpan, relationsToAnnotations);
			List<Pair<TokenSpan, T>> cast = new ArrayList<Pair<TokenSpan, T>>(coref.size() * 2);
			for (Pair<TokenSpan, TokenSpanCluster> corefSpan : coref)
				cast.add(new Pair<TokenSpan, T>(corefSpan.getFirst(), annotationType.getAnnotationClass().cast(corefSpan.getSecond())));
			return cast;
		} else {
			return null;
		}
	}
	
	public <T> T getTokenAnnotation(AnnotationTypeNLP<T> annotationType, int sentenceIndex, int tokenIndex) {	
		if (annotationType.equals(AnnotationTypeNLP.TOKEN)) {
			return annotationType.getAnnotationClass().cast(getToken(sentenceIndex, tokenIndex));
		} else if (annotationType.equals(AnnotationTypeNLP.POS)) {
			return annotationType.getAnnotationClass().cast(getPoSTag(sentenceIndex, tokenIndex));
		} else {
			return null;
		}
		
	}
	
	public DocumentAnnotation toRTWDocumentAnnotation() {
		return toRTWDocumentAnnotation(this.dataTools.getAnnotationTypesNLP());
	}
	
	public boolean fromRTWDocumentAnnotation(DocumentAnnotation documentAnnotation) {
		return fromRTWDocumentAnnotation(documentAnnotation, null);
	}
	
	public Document makeInstanceFromRTWDocumentAnnotation(DocumentAnnotation documentAnnotation) {
		return makeInstanceFromRTWDocumentAnnotation(documentAnnotation, null);
	}
	
	public abstract String getOriginalText();
	public abstract Language getLanguage();
	public abstract int getSentenceCount();
	public abstract int getSentenceTokenCount(int sentenceIndex);
	public abstract String getText();
	public abstract String getSentence(int sentenceIndex);
	public abstract Token getToken(int sentenceIndex, int tokenIndex);
	public abstract PoSTag getPoSTag(int sentenceIndex, int tokenIndex);
	public abstract ConstituencyParse getConstituencyParse(int sentenceIndex);
	public abstract DependencyParse getDependencyParse(int sentenceIndex);
	public abstract List<Pair<TokenSpan, String>> getNer(TokenSpan tokenSpan, TokenSpan.Relation[] relationsToAnnotations);
	public abstract List<Pair<TokenSpan, TokenSpanCluster>> getCoref(TokenSpan tokenSpan, TokenSpan.Relation[] relationsToAnnotations);
	
	public abstract boolean fromRTWDocumentAnnotation(DocumentAnnotation documentAnnotation, Map<AnnotationTypeNLP<?>, String> annotators);
	public abstract Document makeInstanceFromRTWDocumentAnnotation(DocumentAnnotation documentAnnotation, Map<AnnotationTypeNLP<?>, String> annotators);
	public abstract DocumentAnnotation toRTWDocumentAnnotation(Collection<AnnotationTypeNLP<?>> annotationTypes);
}
