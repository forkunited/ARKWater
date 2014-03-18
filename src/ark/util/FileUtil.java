package ark.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

/**
 * 
 * FileUtil helps deal with files on a typical file system or 
 * HDFS (in Hadoop).  Given a path for which to read/write, it
 * determines whether the path points to an HDFS or local
 * file system, and then performs operations accordingly.
 * 
 * @author Bill McDowell
 *
 */
public class FileUtil {
	public static BufferedReader getFileReader(String path) {
		File localFile = new File(path);
		try {
			if (localFile.exists())
				return new BufferedReader(new FileReader(localFile));
		} catch (Exception e) { }
		return HadoopUtil.getFileReader(path);
	}
	
	public static BufferedReader getPropertiesReader(String[] possiblePaths) {
		for (int i = 0; i < possiblePaths.length; i++) {
			BufferedReader r = getFileReader(possiblePaths[i]);
			if (r != null)
				return r;
		}
		return null;
	}
}
