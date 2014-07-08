package ark.data.feature;

import ark.data.annotation.Datum;

/**
 * For datum d, string extractor S, and gazetteer G, 
 * FeatureGazetteerContains computes
 * 
 * max_{g\in G} 1(g=S(d))
 * 
 * @author Bill McDowell
 *
 * @param <D> datum type
 * @param <L> datum label type
 */
public class FeatureGazetteerContains<D extends Datum<L>, L> extends FeatureGazetteer<D, L> { 
	public FeatureGazetteerContains() {
		this.extremumType = FeatureGazetteer.ExtremumType.Maximum;
	}
	
	@Override
	protected double computeExtremum(String str) {
		if (this.gazetteer.contains(str))
			return 1.0;
		else 
			return 0.0;
	}
	
	@Override
	public String getGenericName() {
		return "GazetteerContains";
	}
	
	@Override
	protected Feature<D, L> makeInstance() {
		return new FeatureGazetteerContains<D, L>();
	}
}
