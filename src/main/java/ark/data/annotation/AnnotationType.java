package ark.data.annotation;

import ark.util.JSONSerializable;
import ark.util.StringSerializable;

public class AnnotationType<T> {
	public enum SerializationType {
		ENUM,      // deserialize to enum
		IDENTITY,  // deserialize to string
		JSON,      // deserialize to JSONSerializable
		STRING,    // deserialize to StringSerializable
		OTHER
	}
	
	protected SerializationType serializationType;
	protected String type;
	protected Class<T> annotationClass;
	
	public AnnotationType(String type, Class<T> annotationClass) {
		this.type = type;
		this.annotationClass = annotationClass;
		
		if (JSONSerializable.class.isAssignableFrom(this.annotationClass))
			this.serializationType = SerializationType.JSON;
		else if (StringSerializable.class.isAssignableFrom(this.annotationClass))
			this.serializationType = SerializationType.STRING;
		else if (this.annotationClass.isEnum())
			this.serializationType = SerializationType.ENUM;
		else if (String.class.isAssignableFrom(this.annotationClass))
			this.serializationType = SerializationType.IDENTITY;
		else
			this.serializationType = SerializationType.OTHER;
	}
	
	public Class<T> getAnnotationClass() {
		return this.annotationClass;
	}
	
	public String getType() {
		return this.type;
	}
	
	@Override
	public String toString() {
		return this.type;
	}
	
	@Override
	public boolean equals(Object obj) {
		@SuppressWarnings("rawtypes")
		AnnotationType typeObj = (AnnotationType)obj;
		return this.type.equals(typeObj.type);
	}
	
	@Override
	public int hashCode() {
		return this.type.hashCode();
	}
	
	public SerializationType getSerializationType() {
		return this.serializationType;
	}
}
