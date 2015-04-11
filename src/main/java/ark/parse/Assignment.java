package ark.parse;

import java.io.IOException;
import java.io.Writer;
import java.util.List;

public abstract class Assignment extends Serializable {
	protected String name;
	protected Obj value;
	
	public String getName() {
		return this.name;
	}
	
	public Obj getValue() {
		return this.value;
	}
	
	public abstract boolean isTyped();
	
	public static AssignmentUntyped assignmentUntyped(String name, Obj value) { return new AssignmentUntyped(name, value); }
	public static AssignmentUntyped assignmentUntyped(Obj value) { return new AssignmentUntyped(null, value); }
	public static class AssignmentUntyped extends Assignment {
		public AssignmentUntyped(String name, Obj value) {
			this.name = name;
			this.value = value;
		}
		
		@Override
		public boolean serialize(Writer writer) throws IOException {
			if (this.name != null) {	
				writer.write(this.name);
				writer.write("=");
			}
			
			if (!this.value.serialize(writer))
				return false;
			
			return true;
		}

		@Override
		public boolean isTyped() {
			return false;
		}
	}
	
	public static AssignmentTyped assignmentTyped(List<String> modifiers, String type, String name, Obj value) { return new AssignmentTyped(modifiers, type, name, value); }
	public static class AssignmentTyped extends Assignment {
		private String type;
		private List<String> modifiers;
		
		public AssignmentTyped(List<String> modifiers, String type, String name, Obj value) {
			this.name = name;
			this.value = value;
			this.type = type;
			this.modifiers = modifiers;
		}
		
		public String getType() {
			return this.type;
		}
		
		public List<String> getModifiers() {
			return this.modifiers;
		}
	
		@Override
		public boolean serialize(Writer writer) throws IOException {
			if (this.modifiers != null) {
				for (String modifier : this.modifiers) {
					writer.write(modifier);
					writer.write(" ");
				}
			}
			
			writer.write(this.type);
			writer.write(" ");
			writer.write(this.name);
			writer.write("=");
			
			if (!this.value.serialize(writer))
				return false;
			
			return true;
		}

		@Override
		public boolean isTyped() {
			return true;
		}
	}
}