package ark.model.annotator;

import java.util.HashMap;
import java.util.Map;

import ark.data.annotation.AnnotationType;
import ark.data.annotation.Document;

public abstract class Pipeline {
	protected Map<AnnotationType<?>, Annotator<?>> annotators;
	
	public Pipeline() {
		this.annotators = new HashMap<AnnotationType<?>, Annotator<?>>();
	}
	
	public String getAnnotatorName(AnnotationType<?> annotationType) {
		if (this.annotators.containsKey(annotationType))
			return this.annotators.get(annotationType).getName();
		else 
			return null;
	}
	
	public boolean hasAnnotator(AnnotationType<?> annotationType) {
		return this.annotators.containsKey(annotationType);
	}
	
	public boolean meetsAnnotatorRequirements(AnnotationType<?> annotationType, Document document) {
		if (!hasAnnotator(annotationType))
			return false;
		return document.meetsAnnotatorRequirements(this.annotators.get(annotationType).requires());
	}
}
