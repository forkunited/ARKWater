package ark.data.annotation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import net.sf.json.JSONArray;
import net.sf.json.JSONObject;

import org.jdom.Attribute;
import org.jdom.Element;
import org.jdom.JDOMException;
import org.jdom.input.SAXBuilder;
import org.jdom.output.Format;
import org.jdom.output.XMLOutputter;

import ark.data.annotation.nlp.ConstituencyParse;
import ark.data.annotation.nlp.DependencyParse;
import ark.data.annotation.nlp.PoSTag;
import ark.model.annotator.nlp.NLPAnnotator;
import ark.util.FileUtil;

public class DocumentInMemory extends Document {
	public enum StorageType {
		JSON,
		XML
	}
	
	protected String[][] tokens;
	protected PoSTag[][] posTags;
	protected DependencyParse[] dependencyParses; 
	protected ConstituencyParse[] constituencyParses;
	
	public DocumentInMemory() {
		
	}
	
	public DocumentInMemory(JSONObject json) {
		fromJSON(json);
	}
	
	public DocumentInMemory(Element element) {
		fromXML(element);
	}
	
	public DocumentInMemory(String path, StorageType storageType) {
		if (storageType == StorageType.JSON) {
			BufferedReader r = FileUtil.getFileReader(path);
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
		} else {
			SAXBuilder builder = new SAXBuilder();
			org.jdom.Document document = null;
			try {
				document = builder.build(new File(path));
			} catch (JDOMException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			fromXML(document.getRootElement());
		}
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
				PoSTag posTag = getPoSTag(i, j);
				if (posTag != null)
					posTagsJson.add(posTag.toString());
			}
			sentenceJson.put("tokens", tokensJson);
			sentenceJson.put("posTags", posTagsJson);
			sentenceJson.put("dependencyParse", getDependencyParse(i).toString());
			sentenceJson.put("constituencyParse", getConstituencyParse(i).toString());
			
			sentencesJson.add(sentenceJson);
		}
		json.put("sentences", sentencesJson);
		
		return json;
	}
	
	public Element toXML() {
		Element element = new Element("file");
		int sentenceCount = getSentenceCount();
		
		element.setAttribute("name", this.name);
		element.setAttribute("language", this.language.toString());
		element.setAttribute("nlpAnnotator", this.nlpAnnotator);
		
		for (int i = 0; i < sentenceCount; i++) {
			int sentenceTokenCount = getSentenceTokenCount(i);
			Element entryElement = new Element("entry");
			entryElement.setAttribute("sid", String.valueOf(i));
			entryElement.setAttribute("file", this.name);

			Element sentenceElement = new Element("sentence");
			sentenceElement.addContent(getSentence(i).toString().trim());
			entryElement.addContent(sentenceElement);
			
			Element tokensElement = new Element("tokens");
			for (int j = 0; j < sentenceTokenCount; j++) {
				Element tokenElement = new Element("t");
				tokenElement.addContent("\" \" \"" + getToken(i, j) + "\" \" \"");
				
				PoSTag posTag = getPoSTag(i, j);
				if (posTag != null)
					tokenElement.setAttribute("pos", posTag.toString());
				tokensElement.addContent(tokenElement);
			}
			entryElement.addContent(sentenceElement);
			entryElement.addContent(tokensElement);
			
			Element depsElement = new Element("deps");
			depsElement.addContent(getDependencyParse(i).toString());
			entryElement.addContent(depsElement);
			
			Element parseElement = new Element("parse");
			parseElement.addContent(getConstituencyParse(i).toString());
			entryElement.addContent(parseElement);
			
			element.addContent(entryElement);
		}
		
		return element;
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
	
	public boolean saveToXMLFile(String path) {
		try {
			org.jdom.Document document = new org.jdom.Document();
			document.setRootElement(toXML());
			
			FileOutputStream out = new FileOutputStream(new File(path));
			XMLOutputter outputter = new XMLOutputter(Format.getPrettyFormat());
			
			outputter.output(document, out);
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
		
		return true;
	}
	
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
			JSONArray posTagsJson = sentenceJson.getJSONArray("posTags");
			
			this.tokens[i] = new String[tokensJson.size()];
			for (int j = 0; j < tokensJson.size(); j++)
				this.tokens[i][j] = tokensJson.getString(j);
			
			this.posTags[i] = new PoSTag[posTagsJson.size()];
			for (int j = 0; j < posTagsJson.size(); j++)
				this.posTags[i][j] = PoSTag.valueOf(posTagsJson.getString(j));
			
			this.dependencyParses[i] = DependencyParse.fromString(sentenceJson.getString("dependencyParse"), this, i);
			this.constituencyParses[i] = ConstituencyParse.fromString(sentenceJson.getString("constituencyParse"), this, i);
		}
		
		return true;
	}
	
	@SuppressWarnings("unchecked")
	protected boolean fromXML(Element element) {
		boolean hasName = false;
		boolean hasLanguage = false;
		boolean hasNlpAnnotator = false;
		
		List<Attribute> attributes = (List<Attribute>)element.getAttributes();
		for (Attribute attribute : attributes) {
			if (attribute.getName().equals("name"))
				hasName = true;
			else if (attribute.getName().equals("language"))
				hasLanguage = true;
			else if (attribute.getName().equals("nlpAnnotator"))
				hasNlpAnnotator = true;
		}
		
		if (hasName)
			this.name = element.getAttributeValue("name");
		
		if (hasLanguage)
			this.language = Language.valueOf(element.getAttributeValue("language"));
		else
			this.language = Language.English;
			
		if (hasNlpAnnotator)
			this.nlpAnnotator = element.getAttributeValue("nlpAnnotator");
		
		List<Element> entryElements = (List<Element>)element.getChildren("entry");
		this.tokens = new String[entryElements.size()][];
		this.posTags = new PoSTag[entryElements.size()][];
		this.dependencyParses = new DependencyParse[entryElements.size()];
		this.constituencyParses = new ConstituencyParse[entryElements.size()];
		
		for (Element entryElement : entryElements) {
			int sentenceIndex = Integer.parseInt(entryElement.getAttributeValue("sid"));
			
			Element tokensElement = entryElement.getChild("tokens");
			List<Element> tElements = tokensElement.getChildren("t");
			this.tokens[sentenceIndex] = new String[tElements.size()];
			this.posTags[sentenceIndex] = new PoSTag[tElements.size()];
			for (int j = 0; j < tElements.size(); j++) {
				String tElementText = tElements.get(j).getText();
				int firstQuoteIndex = -1;
				int secondQuoteIndex = tElementText.length();
				for (int i = 0; i < 3; i++) {
					firstQuoteIndex = tElementText.indexOf("\"", firstQuoteIndex+1);
					secondQuoteIndex = tElementText.lastIndexOf("\"", secondQuoteIndex-1);
				}
				
				this.tokens[sentenceIndex][j] = tElementText.substring(firstQuoteIndex + 1, secondQuoteIndex);
				List<Attribute> tAttributes = (List<Attribute>)tElements.get(j).getAttributes();
				for (Attribute attribute : tAttributes)
					if (attribute.getName().equals("pos"))
						this.posTags[sentenceIndex][j] = PoSTag.valueOf(attribute.getValue());
			}
			
			Element depsElement = entryElement.getChild("deps");
			this.dependencyParses[sentenceIndex] = DependencyParse.fromString(depsElement.getText(), this, sentenceIndex);
			
			Element parseElement = entryElement.getChild("parse");
			this.constituencyParses[sentenceIndex] = ConstituencyParse.fromString(parseElement.getText(), this, sentenceIndex);
		}
				
		return true;
	}
}
