package ark.model.annotator.nlp;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.Stack;
import java.util.Map.Entry;

import ark.data.annotation.AnnotationType;
import ark.data.annotation.nlp.AnnotationTypeNLP;
import ark.data.annotation.nlp.ConstituencyParse;
import ark.data.annotation.nlp.DependencyParse;
import ark.data.annotation.nlp.DocumentNLP;
import ark.data.annotation.nlp.PoSTag;
import ark.data.annotation.nlp.Token;
import ark.data.annotation.nlp.TokenSpan;
import ark.data.annotation.nlp.TokenSpanCluster;
import ark.data.annotation.nlp.ConstituencyParse.Constituent;
import ark.data.annotation.nlp.DependencyParse.Dependency;
import ark.data.annotation.nlp.DependencyParse.Node;
import ark.util.Pair;
import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.CorefChain.CorefMention;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;

public class PipelineNLPStanford extends PipelineNLP {
	private StanfordCoreNLP tokenPipeline;
	private StanfordCoreNLP nlpPipeline;
	private int minSentenceAnnotationLength;
	private int maxSentenceAnnotationLength;
	private Annotation annotatedText;
	
	public PipelineNLPStanford() {
		this(0, Integer.MAX_VALUE);
	}
	
	
	public PipelineNLPStanford(int minSentenceAnnotationLength, int maxSentenceAnnotationLength) {
		super();
		this.minSentenceAnnotationLength = minSentenceAnnotationLength;
		this.maxSentenceAnnotationLength = maxSentenceAnnotationLength;
	}
	
	public PipelineNLPStanford(PipelineNLPStanford pipeline) {
		this.document = pipeline.document;
		this.tokenPipeline = pipeline.tokenPipeline;
		this.nlpPipeline = pipeline.nlpPipeline;
		this.minSentenceAnnotationLength = pipeline.minSentenceAnnotationLength;
		this.maxSentenceAnnotationLength = pipeline.maxSentenceAnnotationLength;
		this.annotatedText = pipeline.annotatedText;
	}
	
	public boolean initialize() {
		return initialize(null, null);
	}
	
	public boolean initialize(AnnotationTypeNLP<?> disableFrom) {
		return initialize(disableFrom, null);
	}
	
	public boolean initialize(AnnotationTypeNLP<?> disableFrom, Annotator tokenizer) {
		Properties tokenProps = new Properties();
		Properties props = new Properties();
		
		if (tokenizer != null) {
			if (!tokenizer.requirementsSatisfied().containsAll(Annotator.TOKENIZE_AND_SSPLIT) || tokenizer.requirementsSatisfied().size() != 2)
				return false;
			
			String tokenizerClass = tokenizer.getClass().getName();
			tokenProps.put("customAnnotatorClass.tokenize", tokenizerClass);
		    tokenProps.put("customAnnotatorClass.ssplit", tokenizerClass);
		    props.put("customAnnotatorClass.tokenize", tokenizerClass);
		    props.put("customAnnotatorClass.ssplit", tokenizerClass);
		}
		
		tokenProps.put("annotators", "tokenize, ssplit");
		this.tokenPipeline = new StanfordCoreNLP(tokenProps);
		
		String propsStr = "";
		if (disableFrom == null) {
			propsStr = "tokenize, ssplit, pos, lemma, parse, ner, dcoref";
		} else if (disableFrom.equals(AnnotationTypeNLP.TOKEN)) {
			throw new IllegalArgumentException("Can't disable tokenization");
		} else if (disableFrom.equals(AnnotationTypeNLP.POS)) {
			propsStr = "tokenize, ssplit";
		} else if (disableFrom.equals(AnnotationTypeNLP.CONSTITUENCY_PARSE)) {
			propsStr = "tokenize, ssplit, pos, lemma";
		} else if (disableFrom.equals(AnnotationTypeNLP.DEPENDENCY_PARSE)) {
			propsStr = "tokenize, ssplit, pos, lemma";
		} else if (disableFrom.equals(AnnotationTypeNLP.NER)) {
			propsStr = "tokenize, ssplit, pos, lemma, parse";
		} else if (disableFrom.equals(AnnotationTypeNLP.COREF)) {
			propsStr = "tokenize, ssplit, pos, lemma, parse, ner";
		}

		props.put("annotators", propsStr);
		this.nlpPipeline = new StanfordCoreNLP(props);
		
		this.annotators.clear();
		
		this.annotators.put(AnnotationTypeNLP.TOKEN,  new AnnotatorToken<Token>() {
			public String getName() { return "stanford"; }
			public AnnotationType<Token> produces() { return AnnotationTypeNLP.TOKEN; };
			public AnnotationType<?>[] requires() { return new AnnotationType<?>[] { AnnotationTypeNLP.ORIGINAL_TEXT }; }
			public Token[][] annotate(DocumentNLP document) {
				List<CoreMap> sentences = annotatedText.get(SentencesAnnotation.class);
				Token[][] tokens = new Token[sentences.size()][];
				for(int i = 0; i < sentences.size(); i++) {
					List<CoreLabel> sentenceTokens = sentences.get(i).get(TokensAnnotation.class);
					tokens[i] = new Token[sentenceTokens.size()];
					for (int j = 0; j < sentenceTokens.size(); j++) {
						String word = sentenceTokens.get(j).get(TextAnnotation.class); 
						int charSpanStart = sentenceTokens.get(j).beginPosition();
						int charSpanEnd = sentenceTokens.get(j).endPosition();
						tokens[i][j] = new Token(document, word, charSpanStart, charSpanEnd);
					}
				}
				
				return tokens;
			}
		});
		
		if (disableFrom.equals(AnnotationTypeNLP.POS))
			return true;
		
		this.annotators.put(AnnotationTypeNLP.POS,  new AnnotatorToken<PoSTag>() {
			public String getName() { return "stanford"; }
			public AnnotationType<PoSTag> produces() { return AnnotationTypeNLP.POS; };
			public AnnotationType<?>[] requires() { return new AnnotationType<?>[] { AnnotationTypeNLP.TOKEN }; }
			public PoSTag[][] annotate(DocumentNLP document) {
				List<CoreMap> sentences = annotatedText.get(SentencesAnnotation.class);
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
		});

		if (disableFrom.equals(AnnotationTypeNLP.CONSTITUENCY_PARSE))
			return true;
		
		this.annotators.put(AnnotationTypeNLP.CONSTITUENCY_PARSE,  new AnnotatorSentence<ConstituencyParse>() {
			public String getName() { return "stanford"; }
			public AnnotationType<ConstituencyParse> produces() { return AnnotationTypeNLP.CONSTITUENCY_PARSE; };
			public AnnotationType<?>[] requires() { return new AnnotationType<?>[] { AnnotationTypeNLP.TOKEN, AnnotationTypeNLP.POS }; }
			public Map<Integer, ConstituencyParse> annotate(DocumentNLP document) {
				List<CoreMap> sentences = annotatedText.get(SentencesAnnotation.class);
				Map<Integer, ConstituencyParse> parses = new HashMap<Integer, ConstituencyParse>();
				
				for(int i = 0; i < sentences.size(); i++) {
					Tree tree = sentences.get(i).get(TreeAnnotation.class);

					Constituent root = null;
					parses.put(i, new ConstituencyParse(document, i, null));
					Stack<Pair<Tree, List<Constituent>>> constituents = new Stack<Pair<Tree, List<Constituent>>>();
					Stack<Tree> toVisit = new Stack<Tree>();
					toVisit.push(tree);
					int tokenIndex = 0;
					while (!toVisit.isEmpty()) {
						Tree currentTree = toVisit.pop();
						
						if (!constituents.isEmpty()) {
							while (!isStanfordTreeParent(currentTree, constituents.peek().getFirst())) {
								Pair<Tree, List<Constituent>> currentNeighbor = constituents.pop();
								ConstituencyParse.Constituent constituent = parses.get(i).new Constituent(currentNeighbor.getFirst().label().value(), currentNeighbor.getSecond().toArray(new ConstituencyParse.Constituent[0]));
								constituents.peek().getSecond().add(constituent);
							}
						}
						
						if (currentTree.isPreTerminal()) {
							String label = currentTree.label().value();
							ConstituencyParse.Constituent constituent = parses.get(i).new Constituent(label, new TokenSpan(document, i, tokenIndex, tokenIndex + 1));
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
						root = parses.get(i).new Constituent(possibleRoot.getFirst().label().value(), possibleRoot.getSecond().toArray(new ConstituencyParse.Constituent[0]));
						if (!constituents.isEmpty())
							constituents.peek().getSecond().add(root);
					}
					
					parses.put(i, new ConstituencyParse(document, i, root));
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
		});
		
		if (disableFrom.equals(AnnotationTypeNLP.DEPENDENCY_PARSE))
			return true;
		
		this.annotators.put(AnnotationTypeNLP.DEPENDENCY_PARSE,  new AnnotatorSentence<DependencyParse>() {
			public String getName() { return "stanford"; }
			public AnnotationType<DependencyParse> produces() { return AnnotationTypeNLP.DEPENDENCY_PARSE; };
			public AnnotationType<?>[] requires() { return new AnnotationType<?>[] { AnnotationTypeNLP.TOKEN, AnnotationTypeNLP.POS, AnnotationTypeNLP.CONSTITUENCY_PARSE }; }
			public Map<Integer, DependencyParse> annotate(DocumentNLP document) {
				List<CoreMap> sentences = annotatedText.get(SentencesAnnotation.class);
				Map<Integer, DependencyParse> parses = new HashMap<Integer, DependencyParse>();
				for(int i = 0; i < sentences.size(); i++) {
					SemanticGraph sentenceDependencyGraph = sentences.get(i).get(CollapsedCCProcessedDependenciesAnnotation.class);
					
					Set<IndexedWord> sentenceWords = sentenceDependencyGraph.vertexSet();
					
					Map<Integer, Pair<List<DependencyParse.Dependency>, List<DependencyParse.Dependency>>> nodesToDeps = new HashMap<Integer, Pair<List<DependencyParse.Dependency>, List<DependencyParse.Dependency>>>();
					parses.put(i, new DependencyParse(document, i, null, null));
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
							
							DependencyParse.Dependency dependency = parses.get(i).new Dependency(govIndex, depIndex, relation.getShortName());
							
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
						DependencyParse.Dependency dependency = parses.get(i).new Dependency(-1, depIndex, "root");
						
						if (!nodesToDeps.containsKey(depIndex))
							nodesToDeps.put(depIndex, new Pair<List<Dependency>, List<Dependency>>(new ArrayList<Dependency>(), new ArrayList<Dependency>()));
						
						nodesToDeps.get(-1).getSecond().add(dependency);
						nodesToDeps.get(depIndex).getFirst().add(dependency);
					}
					
					Node[] tokenNodes = new Node[maxIndex+1];
					for (int j = 0; j < tokenNodes.length; j++)
						if (nodesToDeps.containsKey(j))
							tokenNodes[j] = parses.get(i).new Node(j, nodesToDeps.get(j).getFirst().toArray(new Dependency[0]), nodesToDeps.get(j).getSecond().toArray(new Dependency[0]));
					
					Node rootNode = parses.get(i).new Node(-1, new Dependency[0], nodesToDeps.get(-1).getSecond().toArray(new Dependency[0]));
					parses.put(i, new DependencyParse(document, i, rootNode, tokenNodes));
				}
				
				return parses;
			}
		});
	
		if (disableFrom.equals(AnnotationTypeNLP.NER))
			return true;
		
		this.annotators.put(AnnotationTypeNLP.NER,  new AnnotatorTokenSpan<String>() {
			public String getName() { return "stanford"; }
			public AnnotationType<String> produces() { return AnnotationTypeNLP.NER; };
			public AnnotationType<?>[] requires() { return new AnnotationType<?>[] { AnnotationTypeNLP.TOKEN, AnnotationTypeNLP.POS, AnnotationTypeNLP.CONSTITUENCY_PARSE, AnnotationTypeNLP.DEPENDENCY_PARSE }; }
			public List<Pair<TokenSpan, String>> annotate(DocumentNLP document) {
				// FIXME Don't need to do this in a two step process where construct
				// array and then convert it into token span list.  This was just refactored
				// from old code in a rush, but can be done more efficiently
				
				List<CoreMap> sentences = annotatedText.get(SentencesAnnotation.class);
				String[][] ner = new String[sentences.size()][];
				for(int i = 0; i < sentences.size(); i++) {
					List<CoreLabel> sentenceTokens = sentences.get(i).get(TokensAnnotation.class);
					ner[i] = new String[sentenceTokens.size()];
					for (int j = 0; j < sentenceTokens.size(); j++) {
						ner[i][j] = sentenceTokens.get(j).get(NamedEntityTagAnnotation.class); 
					}
				}
				
				List<Pair<TokenSpan, String>> nerAnnotations = new ArrayList<Pair<TokenSpan, String>>();
				for (int i = 0; i < ner.length; i++) {
					for (int j = 0; j < ner[i].length; j++) {
						if (ner[i][j] != null) {
							int endTokenIndex = j + 1;
							for (int k = j + 1; k < ner[i].length; k++) {
								if (ner[i][k] == null || !ner[i][k].equals(ner[i][j])) {
									endTokenIndex = k;
									break;
								}
								ner[i][k] = null;
							}
							
							nerAnnotations.add(new Pair<TokenSpan, String>(new TokenSpan(document, i, j, endTokenIndex), ner[i][j]));
						}
					}
				}
				
				return nerAnnotations;
			}
		});
		
		if (disableFrom.equals(AnnotationTypeNLP.COREF))
			return true;
		
		this.annotators.put(AnnotationTypeNLP.COREF,  new AnnotatorTokenSpan<TokenSpanCluster>() {
			public String getName() { return "stanford"; }
			public AnnotationType<TokenSpanCluster> produces() { return AnnotationTypeNLP.COREF; };
			public AnnotationType<?>[] requires() { return new AnnotationType<?>[] { AnnotationTypeNLP.TOKEN, AnnotationTypeNLP.POS, AnnotationTypeNLP.CONSTITUENCY_PARSE, AnnotationTypeNLP.DEPENDENCY_PARSE, AnnotationTypeNLP.NER }; }
			public List<Pair<TokenSpan, TokenSpanCluster>> annotate(DocumentNLP document) {
				Map<Integer, CorefChain> corefGraph = annotatedText.get(CorefChainAnnotation.class);
				List<Pair<TokenSpan, TokenSpanCluster>> annotations = new ArrayList<Pair<TokenSpan, TokenSpanCluster>>();
				
				for (Entry<Integer, CorefChain> entry : corefGraph.entrySet()) {
					CorefChain corefChain = entry.getValue();
					CorefMention representativeMention = corefChain.getRepresentativeMention();
					TokenSpan representativeSpan = new TokenSpan(document, 
																 representativeMention.sentNum - 1,
																 representativeMention.startIndex - 1,
																 representativeMention.endIndex - 1);
					
					List<TokenSpan> spans = new ArrayList<TokenSpan>();
					Map<IntPair, Set<CorefMention>> mentionMap = corefChain.getMentionMap();
					for (Entry<IntPair, Set<CorefMention>> spanEntry : mentionMap.entrySet()) {
						for (CorefMention mention : spanEntry.getValue()) {
							spans.add(new TokenSpan(document,
														mention.sentNum - 1,
														mention.startIndex - 1,
														mention.endIndex - 1));
						}
					}
					
					TokenSpanCluster cluster = new TokenSpanCluster(entry.getKey(), representativeSpan, spans);
					for (TokenSpan span : spans)
						annotations.add(new Pair<TokenSpan, TokenSpanCluster>(span, cluster));
				}
				
				return annotations;
			}
		});
		
		return true;
	}
	
	
	public boolean setDocument(DocumentNLP document) {
		if (!super.setDocument(document))
			return false;
		
		if (this.tokenPipeline == null) {
			if (!initialize())
				return false;
		}
		
		this.annotatedText = new Annotation(filterText(document.getOriginalText()));
		this.nlpPipeline.annotate(this.annotatedText);
		
		return true;
	}
	
	@SuppressWarnings("unchecked")
	private String filterText(String text) {
		if (this.minSentenceAnnotationLength == 0 && this.maxSentenceAnnotationLength == Integer.MAX_VALUE)
			return text;
		
		Annotation tempAnnotatedText = this.annotatedText;
		this.annotatedText = new Annotation(text);
		this.tokenPipeline.annotate(this.annotatedText);
		Token[][] tokens = ((AnnotatorToken<Token>)this.annotators.get(AnnotationTypeNLP.TOKEN)).annotate(this.document);
		this.annotatedText = tempAnnotatedText;
		
		StringBuilder cleanTextBuilder = new StringBuilder();
		for (int i = 0; i < tokens.length; i++) {
			if (tokens[i].length < this.minSentenceAnnotationLength || tokens[i].length > this.maxSentenceAnnotationLength)
				continue;
			
			int endSymbolsStartToken = tokens[i].length + 1;
			for (int j = tokens[i].length - 1; j >= 0; j--) {
				if (tokens[i][j].getStr().matches("[^A-Za-z0-9]+")) {
					endSymbolsStartToken = j;
				} else {
					break;
				}
			}
			
			for (int j = 0; j < tokens[i].length; j++) {
				cleanTextBuilder.append(tokens[i][j]);
				if (j < endSymbolsStartToken - 1)
					cleanTextBuilder.append(" ");
			}
			
			cleanTextBuilder.append(" ");
		}
		
		return cleanTextBuilder.toString().trim();
	}
	
}
