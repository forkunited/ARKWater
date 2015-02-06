package ark.cluster;

import java.util.ArrayList;
import java.util.List;

import ark.data.annotation.Document;
import ark.data.annotation.nlp.TokenSpan;

public class ClustererTokenSpanPoSTag extends Clusterer<TokenSpan> {
	public ClustererTokenSpanPoSTag() {
		
	}
	
	@Override
	public List<String> getClusters(TokenSpan tokenSpan) {
		List<String> clusters = new ArrayList<String>();
		StringBuilder compoundCluster = new StringBuilder();
		Document document = tokenSpan.getDocument();
		for (int i = tokenSpan.getStartTokenIndex(); i < tokenSpan.getEndTokenIndex(); i++) {
			compoundCluster.append(document.getPoSTag(tokenSpan.getSentenceIndex(), i).toString()).append("_");
		}
	
		if (compoundCluster.length() == 0)
			return clusters;
		
		compoundCluster.delete(compoundCluster.length() - 1, compoundCluster.length());
		clusters.add(compoundCluster.toString());
		return clusters;
	}

	@Override
	public String getName() {
		return "PoSTag";
	}
}
