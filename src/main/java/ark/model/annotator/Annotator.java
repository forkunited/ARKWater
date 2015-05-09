package ark.model.annotator;

import ark.data.annotation.AnnotationType;

public interface Annotator<T> {
	String getName();
	AnnotationType<T> produces();
	AnnotationType<?>[] requires();
}
