package ark.util;

import java.util.SortedMap;

import org.ardverk.collection.PatriciaTrie;
import org.ardverk.collection.StringKeyAnalyzer;
import org.ardverk.collection.Trie;
import org.junit.Test;
import org.junit.Assert;

public class DataStructureTest {
	@Test
	public void testTrie() {
		Trie<String, String> trie = new PatriciaTrie<String, String>(StringKeyAnalyzer.CHAR);
		trie.put("cat", "cat");
		trie.put("dog", "dog");
		trie.put("mouse", "mouse");
		trie.put("mole", "mole");
		
		SortedMap<String, String> prefixed = trie.prefixMap("mo");
		Assert.assertEquals(2, prefixed.size());
		Assert.assertTrue(prefixed.containsKey("mouse"));
		Assert.assertTrue(prefixed.containsKey("mole"));
		
		prefixed = trie.prefixMap("d");
		Assert.assertEquals(1, prefixed.size());
		Assert.assertTrue(prefixed.containsKey("dog"));
	}
}
