package ark.data.annotation.nlp;

import java.lang.reflect.Constructor;

import org.json.JSONObject;

import ark.data.annotation.AnnotationType;
import ark.data.annotation.Document;
import ark.util.JSONSerializable;
import ark.util.StringSerializable;

public class AnnotationTypeNLP<T> extends AnnotationType<T> {
	public static final AnnotationTypeNLP<String> ORIGINAL_TEXT = new AnnotationTypeNLP<String>("text", String.class, Target.DOCUMENT);
	public static final AnnotationTypeNLP<Language> LANGUAGE = new AnnotationTypeNLP<Language>("language", Language.class, Target.DOCUMENT);
	public static final AnnotationTypeNLP<DependencyParse> DEPENDENCY_PARSE = new AnnotationTypeNLP<DependencyParse>("dependency_parse", DependencyParse.class, Target.SENTENCE);
	public static final AnnotationTypeNLP<ConstituencyParse> CONSTITUENCY_PARSE = new AnnotationTypeNLP<ConstituencyParse>("constituency_parse", ConstituencyParse.class, Target.SENTENCE);
	public static final AnnotationTypeNLP<String> NER = new AnnotationTypeNLP<String>("ner", String.class, Target.TOKEN_SPAN);
	public static final AnnotationTypeNLP<TokenSpanCluster> COREF = new AnnotationTypeNLP<TokenSpanCluster>("coref", TokenSpanCluster.class, Target.TOKEN_SPAN);
	public static final AnnotationTypeNLP<PoSTag> POS = new AnnotationTypeNLP<PoSTag>("pos", PoSTag.class, Target.TOKEN);
	public static final AnnotationTypeNLP<Token> TOKEN = new AnnotationTypeNLP<Token>("token", Token.class, Target.TOKEN);
	
	protected static abstract class Serializer<T> {
		protected T makeInstance(Class<T> annotationClass, Document document, int sentenceIndex) {
			T instance = null;
			try {
				if (sentenceIndex >= 0) {
					try {
						Constructor<T> constructor = annotationClass.getConstructor(Document.class, Integer.class);
						
						instance = constructor.newInstance(document, sentenceIndex);
					} catch (NoSuchMethodException e) { }
				}
				
				if (instance == null) {
					try {
						Constructor<T> constructor = annotationClass.getConstructor(Document.class);
						instance = constructor.newInstance(document);
					} catch (NoSuchMethodException e) { }
				}
				
				if (instance == null) {
					try {
						instance = annotationClass.newInstance();
					} catch (InstantiationException e) { 
						return null;
					}
				}
			} catch (Exception e) {
				return null;
			}
			
			return instance;
		}
		
		abstract T deserialize(Document document, int sentenceIndex, Object serializedObj);
		abstract Object serialize(T obj);
	}
	
	protected Serializer<T> enumSerializer = new Serializer<T>() {
		@SuppressWarnings({ "unchecked", "rawtypes" })
		@Override
		T deserialize(Document document, int sentenceIndex, Object serializedObj) {
			return annotationClass.cast(Enum.valueOf((Class<Enum>)annotationClass, serializedObj.toString()));
		}

		@Override
		Object serialize(T obj) {
			return obj.toString();
		}
	};
	
	protected Serializer<T> identitySerializer = new Serializer<T>() {
		@Override
		T deserialize(Document document, int sentenceIndex, Object serializedObj) {
			return annotationClass.cast(serializedObj);
		}

		@Override
		Object serialize(T obj) {
			return obj;
		}
	};

	protected Serializer<T> jsonSerializer = new Serializer<T>() {
		@Override
		T deserialize(Document document, int sentenceIndex, Object serializedObj) {
			T instance = makeInstance(AnnotationTypeNLP.this.annotationClass, document, sentenceIndex);
			if (instance == null)
				return null;
			
			if (!((JSONSerializable)instance).fromJSON((JSONObject)serializedObj))
				return null;
			
			return annotationClass.cast(serializedObj);
		}

		@Override
		Object serialize(T obj) {
			return ((JSONSerializable)obj).toJSON();
		}
	};
	
	protected Serializer<T> stringSerializer = new Serializer<T>() {
		@Override
		T deserialize(Document document, int sentenceIndex, Object serializedObj) {
			T instance = makeInstance(AnnotationTypeNLP.this.annotationClass, document, sentenceIndex);
			if (instance == null)
				return null;
			
			if (!((StringSerializable)instance).fromString(serializedObj.toString()))
				return null;
			
			return annotationClass.cast(serializedObj);
		}

		@Override
		Object serialize(T obj) {
			return ((JSONSerializable)obj).toJSON();
		}
	};
	
	public enum Target {
		DOCUMENT,
		SENTENCE,
		TOKEN_SPAN,
		TOKEN
	}
	
	private Target target;
	private Serializer<T> serializer;

	public AnnotationTypeNLP(String type, Class<T> annotationClass, Target target) {
		super(type, annotationClass);
		this.target = target;
		
		if (this.serializationType == SerializationType.ENUM) 
			this.serializer = this.enumSerializer;
		else if (this.serializationType == SerializationType.IDENTITY)
			this.serializer = this.identitySerializer;
		else if (this.serializationType == SerializationType.JSON)
			this.serializer = this.jsonSerializer;
		else if (this.serializationType == SerializationType.STRING)
			this.serializer = this.stringSerializer;
		else
			throw new IllegalArgumentException("Annotation must be given custom deserializer if annotation type is not an enum, string, JSONSerializable, or StringSerializable");
	}
	
	
	public AnnotationTypeNLP(String type, Class<T> annotationClass, Target target, Serializer<T> serializer) {
		super(type, annotationClass);
		this.target = target;
		this.serializer = serializer;
	}
	
	public Target getTarget() {
		return this.target;
	}
	
	public T deserialize(Document document, Object obj) {
		if (this.target == Target.SENTENCE)
			throw new IllegalArgumentException("Deserialization should be given a sentence index if target is sentence.");
		return this.serializer.deserialize(document, -1, obj);
	}
	
	public T deserialize(Document document, int sentenceIndex, Object obj) {
		return this.serializer.deserialize(document, sentenceIndex, obj);
	}
	
	// FIXME This should really take a T (not an Object), but for now this allows for easy serialization
	// in DocumentNLPInMemory of annotation types where T is a wild card
	public Object serialize(Object obj) {
		return this.serializer.serialize(this.annotationClass.cast(obj));
	}
}
