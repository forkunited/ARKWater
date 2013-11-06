package ark.util;

import java.io.BufferedReader;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;

public abstract class ARKProperties {
	protected Properties properties = null;
	protected Map<String, String> env = null;
	
	public ARKProperties(String[] possiblePaths) {
		try {
			this.properties = new Properties();
			BufferedReader reader = FileUtil.getPropertiesReader(possiblePaths);
			this.properties.load(reader);
			reader.close();
			this.env = System.getenv();
			
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	protected String loadProperty(String property) {
		String propertyValue = this.properties.getProperty(property);
		if (this.env != null) {
			for (Entry<String, String> envEntry : this.env.entrySet())
				propertyValue = propertyValue.replace("${" + envEntry.getKey() + "}", envEntry.getValue());
		}
		return propertyValue;
	}
}
