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

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ark.data.annotation.Document;
import ark.util.Pair;

/**
 * DependencyParse represents a typed dependency parse for a sentence 
 * (http://en.wikipedia.org/wiki/Parse_tree#Dependency-based_parse_trees).  
 * 
 * The 'type' for each dependency in the parse can be any string value.
 * The lack of restriction on the 'type' allows this class to represent 
 * dependencies generated by many different NLP libraries.  For example,
 * it can be used to represent dependencies from Stanford CoreNLP given
 * at:
 * 
 * http://nlp.stanford.edu/software/dependencies_manual.pdf
 * 
 * Or it can be used to represent dependencies from FreeLing given at:
 * 
 * http://devel.cpl.upc.edu/freeling/svn/trunk/doc/grammars/ca+esLABELINGtags 
 * 
 * @author Bill McDowell
 *
 */
public class DependencyParse {
	private static Pattern dependencyPattern = Pattern.compile("(.+)\\((.+)\\-([0-9']+), (.+)\\-([0-9']+)\\)");
	
	public class Node {
		private int tokenIndex;
		private Dependency[] governors;
		private Dependency[] dependents;
		
		public Node(int tokenIndex, Dependency[] governors, Dependency[] dependents) {
			this.tokenIndex = tokenIndex;
			this.governors = governors;
			this.dependents = dependents;
		}
		
		public Dependency[] getGovernors() {
			return this.governors;
		}
		
		public Dependency[] getDependents() {
			return this.dependents;
		}
		
		public int getTokenIndex() {
			return this.tokenIndex;
		}
		
		public boolean isRoot() {
			return this.governors == null && this.tokenIndex < 0;
		}
		
		public boolean isLeaf() {
			return this.dependents == null || this.dependents.length == 0;
		}
	}
	
	public class Dependency {
		private int governingTokenIndex;
		private int dependentTokenIndex;
		private String type;
		
		Dependency() {
			
		}
		
		public Dependency(int governingTokenIndex, int dependentTokenIndex, String type) {
			this.governingTokenIndex = governingTokenIndex;
			this.dependentTokenIndex = dependentTokenIndex;
			this.type = type;
		}
		
		public int getGoverningTokenIndex() {
			return this.governingTokenIndex;
		}
		
		public int getDependentTokenIndex() {
			return this.dependentTokenIndex;
		}
		
		public String getType() {
			return this.type;
		}

		public String toString() {
			return this.type + 
				   "(" 
					+ document.getToken(sentenceIndex, this.governingTokenIndex) + "-" + (this.governingTokenIndex + 1) +
					", " + document.getToken(sentenceIndex, this.dependentTokenIndex) + "-" + (this.dependentTokenIndex + 1) + 
					")";
		}
		
		private Dependency fromString(String str) {
			str = str.trim();
			
			Matcher m = dependencyPattern.matcher(str);
			
			if (!m.matches())
				return null;
			
			String type = m.group(1).trim();
			int governingTokenIndex = Integer.parseInt(m.group(3).replace("'", "").trim());
			int dependentTokenIndex = Integer.parseInt(m.group(5).replace("'", "").trim());
			
			return new Dependency(governingTokenIndex-1, dependentTokenIndex-1, type);
		}
	}
	
	/**
	 * DependencyPath represents a path through a 
	 * DependencyParse.
	 * 
	 * @author Bill McDowell
	 *
	 */
	public class DependencyPath {
		private List<Node> nodes;
		
		DependencyPath(List<Node> nodes) {
			this.nodes = nodes;
		}
		
		public int getTokenLength() {
			return this.nodes.size();
		}
		
		public int getDependencyLength() {
			return this.nodes.size() - 1;
		}
		
		public int getTokenIndex(int index) {
			return this.nodes.get(index).getTokenIndex();
		}
		
		public String getDependencyType(int dependencyIndex) {
			int nodeIndex = this.nodes.get(dependencyIndex).getTokenIndex();
			int nextNodeIndex = this.nodes.get(dependencyIndex + 1).getTokenIndex();
			if (isDependencyGoverningNext(dependencyIndex))
				return getDependency(nodeIndex, nextNodeIndex).getType();
			else
				return getDependency(nextNodeIndex, nodeIndex).getType();
		}
		
		public boolean isGovernedByNext(int index) {
			if (index == this.nodes.size() - 1)
				return false;
			
			for (int i = 0; i < this.nodes.get(index).getGovernors().length; i++)
				if (this.nodes.get(index).getGovernors()[i].getGoverningTokenIndex() == this.nodes.get(index + 1).getTokenIndex())
					return true;
			return false;
		}
		
		public boolean isGoverningNext(int index) {
			if (index == this.nodes.size() - 1)
				return false;
			
			for (int i = 0; i < this.nodes.get(index).getDependents().length; i++)
				if (this.nodes.get(index).getDependents()[i].getDependentTokenIndex() == this.nodes.get(index + 1).getTokenIndex())
					return true;
			return false;
		}
		
		public boolean isDependencyGoverningNext(int dependencyIndex) {
			return isGoverningNext(dependencyIndex);
		}
		
		public boolean isAllGoverning() {
			for (int i = 0; i < getDependencyLength(); i++)
				if (!isDependencyGoverningNext(i))
					return false;
			return true;
		}
		
		public boolean isAllGovernedBy() {
			for (int i = 0; i < getDependencyLength(); i++)
				if (isDependencyGoverningNext(i))
					return false;
			return true;
		}
		
		public String toString() {
			return toString(true);
		}
		
		public String toString(boolean includeTypes) {
			StringBuilder str = new StringBuilder();
			for (int i = 0; i < getDependencyLength(); i++) {
				if (includeTypes)
					str = str.append(getDependencyType(i)).append("-");
				str = str.append((isDependencyGoverningNext(i) ? "G" : "D")).append("/");
			}

			if (str.length() > 0)
				str = str.delete(str.length() - 1, str.length());
			
			return str.toString();
		}
	}

	private Document document;
	private int sentenceIndex;
	private Node root;
	private Node[] tokenNodes;
	
	public DependencyParse(Document document, int sentenceIndex, Node root, Node[] tokenNodes) {
		this.document = document;
		this.sentenceIndex = sentenceIndex;
		this.root = (root != null) ? root : new Node(-1, new Dependency[0], new Dependency[0]);
		this.tokenNodes = tokenNodes; 
	}
	
	public DependencyParse(Document document, int sentenceIndex) {
		this(document, sentenceIndex, null, null);
	}
	
	private Node getNode(int tokenIndex) {
		if (tokenIndex >= this.tokenNodes.length)
			return null;
		if (tokenIndex < 0 || this.tokenNodes.length == 0)
			return this.root;
		return this.tokenNodes[tokenIndex];
	}
	
	public Document getDocument() {
		return this.document;
	}
	
	public int getSentenceIndex() {
		return this.sentenceIndex;
	}
	
	public Node[] getTokenNodes() {
		return this.tokenNodes;
	}
	
	public Dependency getDependency(int sourceTokenIndex, int targetTokenIndex) {
		Node source = getNode(sourceTokenIndex);
		for (int i = 0; i < source.getDependents().length; i++)
			if (source.getDependents()[i].getDependentTokenIndex() == targetTokenIndex)
				return source.getDependents()[i];
		return null;
	}
	
	public DependencyPath getPath(int sourceTokenIndex, int targetTokenIndex) {
		
		Node source = getNode(sourceTokenIndex);
		Node target = getNode(targetTokenIndex);;
		// this can happen when the ccompressed path compresses a node into an arc, and i'm trying to find the path to that node.
		if (source == null || target == null)
			return null;
		
		Stack<Node> toVisit = new Stack<Node>();
		Map<Node, Node> paths = new HashMap<Node, Node>();
		
		toVisit.push(target);
		paths.put(target, null);
		while (!toVisit.isEmpty()) {
			Node current = toVisit.pop();
		
			if (current.equals(source)) {
				List<Node> path = new ArrayList<Node>();
				Node pathCurrent = current;
				while (pathCurrent != null) {
					path.add(pathCurrent);
					pathCurrent = paths.get(pathCurrent);
				}
				return new DependencyPath(path);
			}
			
			Dependency[] governors = current.getGovernors();
			for (Dependency governor : governors) {
				Node governorNode = getNode(governor.getGoverningTokenIndex());
				if (paths.containsKey(governorNode))
					continue;
				toVisit.push(governorNode);
				paths.put(governorNode, current);
			}
			
			Dependency[] dependents = current.getDependents();
			for (Dependency dependent : dependents){
				Node dependentNode = getNode(dependent.getDependentTokenIndex());
				if (paths.containsKey(dependentNode))
					continue;
				toVisit.push(dependentNode);
				paths.put(dependentNode, current);
			}
		}
		
		return null;
	}
	
	public List<Dependency> getGoverningDependencies(int index) {
		Node node = getNode(index);
		List<Dependency> governors = new ArrayList<Dependency>();		
		// FIXME: Temporary fix... node should never be null but it is when the dependency 
		// parse is empty sometimes for unknown reasons
		if (node == null)
			return governors;

		for (int i = 0; i < node.getGovernors().length; i ++)
			governors.add(node.getGovernors()[i]);
			
		return governors;
	}
	
	public List<Dependency> getGovernedDependencies(int index) {
		Node node = getNode(index);
		List<Dependency> dependencies = new ArrayList<Dependency>();
		// FIXME: Temporary fix... node should never be null but it is when the dependency 
		// parse is empty sometimes for unknown reasons
		if (node == null)
			return dependencies;
		
		for (int i = 0; i < node.getDependents().length; i ++)
			dependencies.add(node.getDependents()[i]);
			
		return dependencies;
	}
	
	public String toString() {
		if (this.root == null)
			return "";
		
		StringBuilder str = new StringBuilder();
		
		for (int i = 0; i < this.tokenNodes.length; i++)
			if (this.tokenNodes[i] != null) {
				for (int j = 0; j < this.tokenNodes[i].getGovernors().length; j++)
					str = str.append(this.tokenNodes[i].getGovernors()[j].toString()).append("\n");
			}
		
		return str.toString();
	}
	
	public static DependencyParse fromString(String str, Document document, int sentenceIndex) {
		if (str.trim().length() == 0)
			return new DependencyParse(document, sentenceIndex);
		
		String[] strParts = str.split("\n");
		Map<Integer, Pair<List<Dependency>, List<Dependency>>> nodesToDeps = new HashMap<Integer, Pair<List<Dependency>, List<Dependency>>>();
		DependencyParse parse = new DependencyParse(document, sentenceIndex, null, null);

		int maxIndex = -1;
		for (int i = 0; i < strParts.length; i++) {
			Dependency dependency = (parse.new Dependency()).fromString(strParts[i]);
			if (dependency == null)
				return new DependencyParse(document, sentenceIndex);
			
			int govIndex = dependency.getGoverningTokenIndex();
			int depIndex = dependency.getDependentTokenIndex();
			
			maxIndex = Math.max(depIndex, Math.max(govIndex, maxIndex));
			
			if (!nodesToDeps.containsKey(govIndex))
				nodesToDeps.put(govIndex, new Pair<List<Dependency>, List<Dependency>>(new ArrayList<Dependency>(), new ArrayList<Dependency>()));
			if (!nodesToDeps.containsKey(depIndex))
				nodesToDeps.put(depIndex, new Pair<List<Dependency>, List<Dependency>>(new ArrayList<Dependency>(), new ArrayList<Dependency>()));
			
			nodesToDeps.get(govIndex).getSecond().add(dependency);
			nodesToDeps.get(depIndex).getFirst().add(dependency);
		}
		
		Node[] tokenNodes = new Node[maxIndex+1];
		for (int i = 0; i < tokenNodes.length; i++)
			if (nodesToDeps.containsKey(i))
				tokenNodes[i] = parse.new Node(i, nodesToDeps.get(i).getFirst().toArray(new Dependency[0]), nodesToDeps.get(i).getSecond().toArray(new Dependency[0]));

		if (!nodesToDeps.containsKey(-1)) {
			throw new IllegalArgumentException("Failed to get root for " + document.getName() + " " + sentenceIndex);
		}
			
		parse.root =  parse.new Node(-1, new Dependency[0], nodesToDeps.get(-1).getSecond().toArray(new Dependency[0]));
		parse.tokenNodes = tokenNodes;
				
		return parse;
	}
	
	public DependencyParse clone(Document document) {
		return new DependencyParse(document, this.sentenceIndex, this.root, this.tokenNodes);
	}
}
