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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.Stack;

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
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;

import ark.data.annotation.Document;
import ark.data.annotation.Language;
import ark.data.annotation.nlp.ConstituencyParse;
import ark.data.annotation.nlp.DependencyParse;
import ark.data.annotation.nlp.PoSTag;
import ark.data.annotation.nlp.TokenSpan;
import ark.data.annotation.nlp.ConstituencyParse.Constituent;
import ark.data.annotation.nlp.DependencyParse.Dependency;
import ark.data.annotation.nlp.DependencyParse.Node;
import ark.util.Pair;

/**
 * NLPAnnotatorStanford supplements a text with NLP 
 * annotations (parses, part-of-speech tags, etc) using 
 * the Stanford CoreNLP pipeline
 * (http://nlp.stanford.edu/software/corenlp.shtml)
 * 
 * The returned NLP annotations are in the ARKWater 
 * (https://github.com/forkunited/ARKWater)
 * library's format.
 * 
 * Once constructed for a specified language, the annotator 
 * object can be used by calling
 * the setText method to set the text to be annotated, and
 * then calling the make[X] methods to retrieve the annotations
 * for that text.
 * 
 * @author Bill McDowell
 *
 */
public class NLPAnnotatorStanford extends NLPAnnotator {
	private StanfordCoreNLP pipeline;
	private Annotation annotatedText;
	
	public NLPAnnotatorStanford() {
		setLanguage(Language.English);
	}
	
	/**
	 * @param language
	 * @return true if the language of the annotator is successfully 
	 * set.  Currently, the only supported language is English.
	 */
	public boolean setLanguage(Language language) {
		if (language != Language.English)
			return false;
		this.language = language;
		return true;
	}
	
	public String toString() {
		return "Stanford";
	}
	
	/**
	 * @param text
	 * @return true if the annotator has received the text 
	 * and is ready to return annotations for it
	 */
	public boolean setText(String text) {
		if (this.pipeline == null) {
			Properties props = new Properties();
			String propsStr = "tokenize, ssplit";
			if (!this.disabledPoSTags)
				propsStr = propsStr + ", pos";
			if (!this.disabledConstituencyParses || !this.disabledDependencyParses)
				propsStr = propsStr + ", lemma, parse";
		    props.put("annotators", propsStr); // ner before parse?
		    this.pipeline = new StanfordCoreNLP(props);
		}
		
		this.text = text;
		this.annotatedText = new Annotation(text);
		this.pipeline.annotate(this.annotatedText);
		
		return true;
	}
	
	/**
	 * @return an array of tokens for each segmented 
	 * sentence of the text.
	 */
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
	
	/**
	 * @return an array of pos-tags for each segmented 
	 * sentence of the text.
	 */
	protected PoSTag[][] makePoSTagsInternal() {
		List<CoreMap> sentences = this.annotatedText.get(SentencesAnnotation.class);
		PoSTag[][] posTags = new PoSTag[sentences.size()][];
		for (int i = 0; i < sentences.size(); i++) {
			List<CoreLabel> sentenceTokens = sentences.get(i).get(TokensAnnotation.class);
			posTags[i] = new PoSTag[sentenceTokens.size()];
			for (int j = 0; j < sentenceTokens.size(); j++) {
				String pos = sentenceTokens.get(j).get(PartOfSpeechAnnotation.class);  
				
				if (pos.length() > 0 && !Character.isLetter(pos.toCharArray()[0]))
					posTags[i][j] = PoSTag.SYM;
				else
					posTags[i][j] = PoSTag.valueOf(pos);
			}
		}
		
		return posTags;
	}
	
	/**
	 * @return a dependency parse for each sentence of text
	 */
	protected DependencyParse[] makeDependencyParsesInternal(Document document, int sentenceIndexOffset) {
		List<CoreMap> sentences = this.annotatedText.get(SentencesAnnotation.class);
		DependencyParse[] parses = new DependencyParse[sentences.size()];
		for(int i = 0; i < sentences.size(); i++) {
			SemanticGraph sentenceDependencyGraph = sentences.get(i).get(CollapsedCCProcessedDependenciesAnnotation.class);
			
			Set<IndexedWord> sentenceWords = sentenceDependencyGraph.vertexSet();
			
			Map<Integer, Pair<List<DependencyParse.Dependency>, List<DependencyParse.Dependency>>> nodesToDeps = new HashMap<Integer, Pair<List<DependencyParse.Dependency>, List<DependencyParse.Dependency>>>();
			parses[i] = new DependencyParse(document, sentenceIndexOffset + i, null, null);
			int maxIndex = -1;
			for (IndexedWord sentenceWord1 : sentenceWords) {
				for (IndexedWord sentenceWord2 : sentenceWords) {
					if (sentenceWord1.equals(sentenceWord2))
						continue;
					GrammaticalRelation relation = sentenceDependencyGraph.reln(sentenceWord1, sentenceWord2);
					if (relation == null)
						continue;
				
					int govIndex = sentenceWord1.index() - 1;
					int depIndex = sentenceWord2.index() - 1;
					
					maxIndex = Math.max(depIndex, Math.max(govIndex, maxIndex));
					
					DependencyParse.Dependency dependency = parses[i].new Dependency(govIndex, depIndex, relation.getShortName());
					
					if (!nodesToDeps.containsKey(govIndex))
						nodesToDeps.put(govIndex, new Pair<List<Dependency>, List<Dependency>>(new ArrayList<Dependency>(), new ArrayList<Dependency>()));
					if (!nodesToDeps.containsKey(depIndex))
						nodesToDeps.put(depIndex, new Pair<List<Dependency>, List<Dependency>>(new ArrayList<Dependency>(), new ArrayList<Dependency>()));
					
					nodesToDeps.get(govIndex).getSecond().add(dependency);
					nodesToDeps.get(depIndex).getFirst().add(dependency);
				}
			}
			
			if (!nodesToDeps.containsKey(-1))
				nodesToDeps.put(-1, new Pair<List<Dependency>, List<Dependency>>(new ArrayList<Dependency>(), new ArrayList<Dependency>()));
			
			
			Collection<IndexedWord> rootDeps = sentenceDependencyGraph.getRoots();
			for (IndexedWord rootDep : rootDeps) {
				int depIndex = rootDep.index() - 1;
				DependencyParse.Dependency dependency = parses[i].new Dependency(-1, depIndex, "root");
				
				if (!nodesToDeps.containsKey(depIndex))
					nodesToDeps.put(depIndex, new Pair<List<Dependency>, List<Dependency>>(new ArrayList<Dependency>(), new ArrayList<Dependency>()));
				
				nodesToDeps.get(-1).getSecond().add(dependency);
				nodesToDeps.get(depIndex).getFirst().add(dependency);
			}
			
			Node[] tokenNodes = new Node[maxIndex+1];
			for (int j = 0; j < tokenNodes.length; j++)
				if (nodesToDeps.containsKey(j))
					tokenNodes[j] = parses[i].new Node(j, nodesToDeps.get(j).getFirst().toArray(new Dependency[0]), nodesToDeps.get(j).getSecond().toArray(new Dependency[0]));
			
			Node rootNode = parses[i].new Node(-1, new Dependency[0], nodesToDeps.get(-1).getSecond().toArray(new Dependency[0]));
			parses[i] = new DependencyParse(document, sentenceIndexOffset + i, rootNode, tokenNodes);
		}
		
		return parses;
	}
	
	/**
	 * @return a constituency parse for each sentence of text
	 */
	protected ConstituencyParse[] makeConstituencyParsesInternal(Document document, int sentenceIndexOffset) {
		List<CoreMap> sentences = this.annotatedText.get(SentencesAnnotation.class);
		ConstituencyParse[] parses = new ConstituencyParse[sentences.size()];
		
		for(int i = 0; i < sentences.size(); i++) {
			Tree tree = sentences.get(i).get(TreeAnnotation.class);

			Constituent root = null;
			parses[i] = new ConstituencyParse(document, sentenceIndexOffset + i, null);
			Stack<Pair<Tree, List<Constituent>>> constituents = new Stack<Pair<Tree, List<Constituent>>>();
			Stack<Tree> toVisit = new Stack<Tree>();
			toVisit.push(tree);
			int tokenIndex = 0;
			while (!toVisit.isEmpty()) {
				Tree currentTree = toVisit.pop();
				
				if (!constituents.isEmpty()) {
					while (!isStanfordTreeParent(currentTree, constituents.peek().getFirst())) {
						Pair<Tree, List<Constituent>> currentNeighbor = constituents.pop();
						ConstituencyParse.Constituent constituent = parses[i].new Constituent(currentNeighbor.getFirst().label().value(), currentNeighbor.getSecond().toArray(new ConstituencyParse.Constituent[0]));
						constituents.peek().getSecond().add(constituent);
					}
				}
				
				if (currentTree.isPreTerminal()) {
					String label = currentTree.label().value();
					ConstituencyParse.Constituent constituent = parses[i].new Constituent(label, new TokenSpan(document, i, tokenIndex, tokenIndex + 1));
					tokenIndex++;
					if (!constituents.isEmpty())
						constituents.peek().getSecond().add(constituent);
					else
						root = constituent;
				} else {
					constituents.push(new Pair<Tree, List<Constituent>>(currentTree, new ArrayList<Constituent>()));
					for (int j = currentTree.numChildren() - 1; j >= 0; j--)
						toVisit.push(currentTree.getChild(j));
				}
			}
			
			while (!constituents.isEmpty()) {
				Pair<Tree, List<Constituent>> possibleRoot = constituents.pop();
				root = parses[i].new Constituent(possibleRoot.getFirst().label().value(), possibleRoot.getSecond().toArray(new ConstituencyParse.Constituent[0]));
				if (!constituents.isEmpty())
					constituents.peek().getSecond().add(root);
			}
			
			parses[i] = new ConstituencyParse(document, sentenceIndexOffset + i, root);
		}
		
		return parses;
	}
	
	private boolean isStanfordTreeParent(Tree tree, Tree possibleParent) {
		for (int j = 0; j < possibleParent.numChildren(); j++) {
			if (possibleParent.getChild(j).equals(tree)) {
				return true;
			}
		}
		return false;
	}
}
