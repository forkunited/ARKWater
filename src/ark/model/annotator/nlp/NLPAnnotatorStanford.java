package ark.model.annotator.nlp;

import java.util.LinkedList;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.util.CoreMap;

import ark.data.annotation.Language;
import ark.data.annotation.nlp.PoSTag;
import ark.data.annotation.nlp.TypedDependency;

public class NLPAnnotatorStanford extends NLPAnnotator {
	private StanfordCoreNLP pipeline;
	private Annotation annotatedText;
	
	public NLPAnnotatorStanford() {
		setLanguage(Language.English);
		
		Properties props = new Properties();
	    props.put("annotators", "tokenize, ssplit, pos, lemma, parse"); // ner before parse?
	    this.pipeline = new StanfordCoreNLP(props);
	}
	
	public boolean setLanguage(Language language) {
		if (language != Language.English)
			return false;
		this.language = language;
		return true;
	}
	
	public String toString() {
		return "Stanford";
	}
	
	public boolean setText(String text) {
		this.text = text;
		this.annotatedText = new Annotation(text);
	    this.pipeline.annotate(this.annotatedText);
		
		return true;
	}
	
	public String[][] makeTokens() {
		List<CoreMap> sentences = this.annotatedText.get(SentencesAnnotation.class);
		String[][] tokens = new String[sentences.size()][];
		for(int i = 0; i < sentences.size(); i++) {
			List<CoreLabel> sentenceTokens = sentences.get(i).get(TokensAnnotation.class);
			tokens[i] = new String[sentenceTokens.size()];
			for (int j = 0; j < sentenceTokens.size(); j++) {
				String word = sentenceTokens.get(j).get(TextAnnotation.class); 
				tokens[i][j] = word;
			}
		}
		
		return tokens;
	}
	
	public PoSTag[][] makePoSTags() {
		List<CoreMap> sentences = this.annotatedText.get(SentencesAnnotation.class);
		PoSTag[][] posTags = new PoSTag[sentences.size()][];
		for(int i = 0; i < sentences.size(); i++) {
			List<CoreLabel> sentenceTokens = sentences.get(i).get(TokensAnnotation.class);
			posTags[i] = new PoSTag[sentenceTokens.size()];
			for (int j = 0; j < sentenceTokens.size(); j++) {
				String pos = sentenceTokens.get(j).get(PartOfSpeechAnnotation.class);  
				
				if (pos.length() > 0 && !Character.isAlphabetic(pos.toCharArray()[0]))
					posTags[i][j] = PoSTag.SYM;
				else
					posTags[i][j] = PoSTag.valueOf(pos);
			}
		}
		
		return posTags;
	}
	
	public TypedDependency[][] makeDependencies() {
		List<CoreMap> sentences = this.annotatedText.get(SentencesAnnotation.class);
		TypedDependency[][] dependencies = new TypedDependency[sentences.size()][];
		for(int i = 0; i < sentences.size(); i++) {
			SemanticGraph sentenceDependencyGraph = sentences.get(i).get(CollapsedCCProcessedDependenciesAnnotation.class);
			Set<IndexedWord> sentenceWords = sentenceDependencyGraph.vertexSet();
			List<TypedDependency> sentenceDependencies = new LinkedList<TypedDependency>();
			
			List<IndexedWord> sentenceRoots = new LinkedList<IndexedWord>();
			sentenceRoots.addAll(sentenceDependencyGraph.getRoots());
			for (IndexedWord sentenceRoot : sentenceRoots)
				sentenceDependencies.add(new TypedDependency(null,
															i,
															-1,
															sentenceRoot.index() - 1,
															"root"));
			
			for (IndexedWord sentenceWord1 : sentenceWords) {
				for (IndexedWord sentenceWord2 : sentenceWords) {
					if (sentenceWord1.equals(sentenceWord2))
						continue;
					GrammaticalRelation relation = sentenceDependencyGraph.reln(sentenceWord1, sentenceWord2);
					if (relation == null)
						continue;
					sentenceDependencies.add(new TypedDependency(null, 
																i, 
																sentenceWord1.index() - 1, 
																sentenceWord2.index() - 1, 
																relation.getShortName()));
				}
			}
			
			dependencies[i] = new TypedDependency[sentenceDependencies.size()];
			dependencies[i] = sentenceDependencies.toArray(dependencies[i]);
		}
		
		return dependencies;
	}
}
