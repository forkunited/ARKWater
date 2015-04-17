package ark.data.feature.fn;


import java.util.Arrays;
import java.util.List;

import org.junit.Test;
import org.junit.Assert;

import ark.data.Context;
import ark.data.DataTools;
import ark.data.annotation.Document;
import ark.data.annotation.DocumentInMemory;
import ark.data.annotation.Language;
import ark.data.annotation.TestDatum;
import ark.data.annotation.nlp.PoSTag;
import ark.data.annotation.nlp.TokenSpan;
import ark.model.annotator.nlp.NLPAnnotatorStanford;
import ark.parse.Obj;
import ark.util.OutputWriter;

public class FnTest {
	private TestDatum<String> testDatum = constructTestDatum();
	private Context<TestDatum<String>, String> context = Context.deserialize(TestDatum.getStringTools(new DataTools(new OutputWriter())),
															"ts_fn head=Head();\n" +
															"ts_fn ins1=NGramInside(n=\"1\", noHead=\"true\");\n" +
															"ts_fn ins2=NGramInside(n=\"2\", noHead=\"true\");\n" +
															"ts_fn ins3=NGramInside(n=\"3\", noHead=\"true\");\n" +
															"ts_fn ctxb1=NGramContext(n=\"1\", type=\"BEFORE\");\n" +
															"ts_fn ctxa1=NGramContext(n=\"1\", type=\"AFTER\");\n" +
															"ts_fn sent1=NGramSentence(n=\"1\", noSpan=\"true\");\n" +
															"ts_fn doc2=NGramDocument(n=\"2\", noSentence=\"true\");\n" +
															"ts_str_fn pos=PoS();\n" +
															"ts_str_fn str=String(cleanFn=\"DefaultCleanFn\");\n" +
															"str_fn pre=Affix(type=\"PREFIX\", n=\"3\");\n" +
															"str_fn suf=Affix(type=\"SUFFIX\", n=\"3\");\n" +
															"str_fn filter=Filter(filter=\"some\", type=\"SUBSTRING\");\n" +
															"str_fn filter_s=Filter(filter=\"some\", type=\"SUFFIX\");\n" +
															"str_fn filter_p=Filter(filter=\"some\", type=\"PREFIX\");\n" +
															"ts_str_fn headDoc2=(${str} o ${head} o ${doc2});\n");
	
	
	private TestDatum<String> constructTestDatum() {
		Document testDocument = new DocumentInMemory("",
													 "This is some test text.  It is good text for testing.  Test it.", 
													 Language.English, new NLPAnnotatorStanford());
		
	
		return new TestDatum<String>(1, new TokenSpan(testDocument, 1, 1, 4), ""); // Refers to "is good text"
	}
	
	@Test
	public void testFnString() {
		FnString str = (FnString)this.context.getMatchTokenSpanStrFn(Obj.curlyBracedValue("str"));
		List<TokenSpan> datumSpan = Arrays.asList(this.context.getDatumTools().getTokenSpanExtractor("TokenSpan").extract(this.testDatum));
		List<String> datumStr = str.compute(datumSpan);
		
		Assert.assertEquals(1, datumStr.size());
		Assert.assertEquals("is_good_text", datumStr.get(0));
	}
	
	@Test 
	public void testFnNGramInside() {
		FnString str = (FnString)this.context.getMatchTokenSpanStrFn(Obj.curlyBracedValue("str"));
		FnNGramInside ins1 = (FnNGramInside)this.context.getMatchTokenSpanFn(Obj.curlyBracedValue("ins1"));
		FnNGramInside ins2 = (FnNGramInside)this.context.getMatchTokenSpanFn(Obj.curlyBracedValue("ins2"));
		FnNGramInside ins3 = (FnNGramInside)this.context.getMatchTokenSpanFn(Obj.curlyBracedValue("ins3"));
	
		List<TokenSpan> datumSpan = Arrays.asList(this.context.getDatumTools().getTokenSpanExtractor("TokenSpan").extract(this.testDatum));
		List<String> ins1Strs = str.compute(ins1.compute(datumSpan));
		List<String> ins2Strs = str.compute(ins2.compute(datumSpan));
		List<String> ins3Strs = str.compute(ins3.compute(datumSpan));
		
		Assert.assertEquals(2, ins1Strs.size());
		Assert.assertEquals(1, ins2Strs.size());
		Assert.assertEquals(0, ins3Strs.size());
		
		Assert.assertEquals("is", ins1Strs.get(0));
		Assert.assertEquals("good", ins1Strs.get(1));
		Assert.assertEquals("is_good", ins2Strs.get(0));
	}
	
	@Test
	public void testFnNGramContext() {
		FnString str = (FnString)this.context.getMatchTokenSpanStrFn(Obj.curlyBracedValue("str"));
		FnNGramContext ctxb1 = (FnNGramContext)this.context.getMatchTokenSpanFn(Obj.curlyBracedValue("ctxb1"));
		FnNGramContext ctxa1 = (FnNGramContext)this.context.getMatchTokenSpanFn(Obj.curlyBracedValue("ctxa1"));
	
		List<TokenSpan> datumSpan = Arrays.asList(this.context.getDatumTools().getTokenSpanExtractor("TokenSpan").extract(this.testDatum));
		List<String> ctxb1Strs = str.compute(ctxb1.compute(datumSpan));
		List<String> ctxa1Strs = str.compute(ctxa1.compute(datumSpan));
		
		Assert.assertEquals(1, ctxb1Strs.size());
		Assert.assertEquals(1, ctxa1Strs.size());
		
		Assert.assertEquals("it", ctxb1Strs.get(0));
		Assert.assertEquals("for", ctxa1Strs.get(0));
	}
	
	@Test
	public void testFnNGramSentence() {
		FnString str = (FnString)this.context.getMatchTokenSpanStrFn(Obj.curlyBracedValue("str"));
		FnNGramSentence sent1 = (FnNGramSentence)this.context.getMatchTokenSpanFn(Obj.curlyBracedValue("sent1"));

		List<TokenSpan> datumSpan = Arrays.asList(this.context.getDatumTools().getTokenSpanExtractor("TokenSpan").extract(this.testDatum));
		List<String> sent1Strs = str.compute(sent1.compute(datumSpan));
		
		Assert.assertEquals(4, sent1Strs.size());
		
		Assert.assertEquals("it", sent1Strs.get(0));
		Assert.assertEquals("for", sent1Strs.get(1));
		Assert.assertEquals("testing", sent1Strs.get(2));
		Assert.assertEquals("", sent1Strs.get(3));
	}
	
	@Test
	public void testFnNGramDocument() {
		FnString str = (FnString)this.context.getMatchTokenSpanStrFn(Obj.curlyBracedValue("str"));
		FnNGramDocument doc2 = (FnNGramDocument)this.context.getMatchTokenSpanFn(Obj.curlyBracedValue("doc2"));

		List<TokenSpan> datumSpan = Arrays.asList(this.context.getDatumTools().getTokenSpanExtractor("TokenSpan").extract(this.testDatum));
		List<String> doc2Strs = str.compute(doc2.compute(datumSpan));
		
		Assert.assertEquals(7, doc2Strs.size());
		
		Assert.assertEquals("this_is", doc2Strs.get(0));
		Assert.assertEquals("is_some", doc2Strs.get(1));
		Assert.assertEquals("some_test", doc2Strs.get(2));
		Assert.assertEquals("test_text", doc2Strs.get(3));
		Assert.assertEquals("text_", doc2Strs.get(4));
		Assert.assertEquals("test_it", doc2Strs.get(5));
		Assert.assertEquals("it_", doc2Strs.get(6));
	}
	
	@Test
	public void testFnHead() {
		FnString str = (FnString)this.context.getMatchTokenSpanStrFn(Obj.curlyBracedValue("str"));
		FnHead head = (FnHead)this.context.getMatchTokenSpanFn(Obj.curlyBracedValue("head"));

		List<TokenSpan> datumSpan = Arrays.asList(this.context.getDatumTools().getTokenSpanExtractor("TokenSpan").extract(this.testDatum));
		List<String> headStrs = str.compute(head.compute(datumSpan));
		
		Assert.assertEquals(1, headStrs.size());	
		Assert.assertEquals("text", headStrs.get(0));
	}
	
	@Test
	public void testFnAffix() {
		FnAffix pre = (FnAffix)this.context.getMatchStrFn(Obj.curlyBracedValue("pre"));
		FnAffix suf = (FnAffix)this.context.getMatchStrFn(Obj.curlyBracedValue("suf"));
		FnString str = (FnString)this.context.getMatchTokenSpanStrFn(Obj.curlyBracedValue("str"));
		FnNGramSentence sent1 = (FnNGramSentence)this.context.getMatchTokenSpanFn(Obj.curlyBracedValue("sent1"));

		List<TokenSpan> datumSpan = Arrays.asList(this.context.getDatumTools().getTokenSpanExtractor("TokenSpan").extract(this.testDatum));
		
		List<String> prefixes = pre.compute(str.compute(sent1.compute(datumSpan)));
		List<String> suffixes = suf.compute(str.compute(sent1.compute(datumSpan)));
		Assert.assertEquals(1, prefixes.size());
		Assert.assertEquals(1, suffixes.size());
		
		Assert.assertEquals("tes", prefixes.get(0));
		Assert.assertEquals("ing", suffixes.get(0));
	}
	
	@Test
	public void testFnFilter() {
		FnString str = (FnString)this.context.getMatchTokenSpanStrFn(Obj.curlyBracedValue("str"));
		FnNGramDocument doc2 = (FnNGramDocument)this.context.getMatchTokenSpanFn(Obj.curlyBracedValue("doc2"));
		FnFilter filter = (FnFilter)this.context.getMatchStrFn(Obj.curlyBracedValue("filter"));
		FnFilter filter_s = (FnFilter)this.context.getMatchStrFn(Obj.curlyBracedValue("filter_s"));
		FnFilter filter_p = (FnFilter)this.context.getMatchStrFn(Obj.curlyBracedValue("filter_p"));
		
		List<TokenSpan> datumSpan = Arrays.asList(this.context.getDatumTools().getTokenSpanExtractor("TokenSpan").extract(this.testDatum));
		List<String> doc2Strs = str.compute(doc2.compute(datumSpan));
		List<String> filterStrs = filter.compute(doc2Strs);
		List<String> filter_sStrs = filter_s.compute(doc2Strs);
		List<String> filter_pStrs = filter_p.compute(doc2Strs);	
		
		Assert.assertEquals(2, filterStrs.size());
		Assert.assertEquals(1, filter_sStrs.size());
		Assert.assertEquals(1, filter_pStrs.size());
		
		Assert.assertEquals("is_some", filterStrs.get(0));
		Assert.assertEquals("some_test", filterStrs.get(1));
		Assert.assertEquals("is_some", filter_sStrs.get(0));
		Assert.assertEquals("some_test", filter_pStrs.get(0));
	}
	
	@Test
	public void testFnPoS() {
		FnPoS pos = (FnPoS)this.context.getMatchTokenSpanStrFn(Obj.curlyBracedValue("pos"));
		FnNGramSentence sent1 = (FnNGramSentence)this.context.getMatchTokenSpanFn(Obj.curlyBracedValue("sent1"));

		List<TokenSpan> datumSpan = Arrays.asList(this.context.getDatumTools().getTokenSpanExtractor("TokenSpan").extract(this.testDatum));
		List<String> sent1PoS = pos.compute(sent1.compute(datumSpan));
	
		Assert.assertEquals(4, sent1PoS.size());
		
		Assert.assertEquals(PoSTag.PRP.toString(), sent1PoS.get(0));
		Assert.assertEquals(PoSTag.IN.toString(),  sent1PoS.get(1));
		Assert.assertEquals(PoSTag.NN.toString(), sent1PoS.get(2));
		Assert.assertEquals(PoSTag.SYM.toString(), sent1PoS.get(3));
	}
	
	@Test
	public void testFnComposite() {
		Fn<List<TokenSpan>, List<String>> headDoc2 = this.context.getMatchTokenSpanStrFn(Obj.curlyBracedValue("headDoc2"));
		List<TokenSpan> datumSpan = Arrays.asList(this.context.getDatumTools().getTokenSpanExtractor("TokenSpan").extract(this.testDatum));
		List<String> headDoc2Strs = headDoc2.compute(datumSpan);
		
		Assert.assertEquals(7, headDoc2Strs.size());
		
		Assert.assertEquals("is", headDoc2Strs.get(0));
		Assert.assertEquals("some", headDoc2Strs.get(1));
		Assert.assertEquals("test", headDoc2Strs.get(2));
		Assert.assertEquals("text", headDoc2Strs.get(3));
		Assert.assertEquals("", headDoc2Strs.get(4));
		Assert.assertEquals("it", headDoc2Strs.get(5));
		Assert.assertEquals("", headDoc2Strs.get(6));
	}
}
