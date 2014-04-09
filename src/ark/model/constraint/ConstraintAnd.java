package ark.model.constraint;

import ark.data.annotation.Datum;
import ark.data.feature.FeaturizedDataSet;

public class ConstraintAnd<D extends Datum<L>, L> extends Constraint<D, L> {
	private Constraint<D, L> firstConstraint;
	private Constraint<D, L> secondConstraint;
	
	public ConstraintAnd(Constraint<D,L> firstConstraint, Constraint<D, L> secondConstraint) {
		this.firstConstraint = firstConstraint;
		this.secondConstraint = secondConstraint;
	}

	@Override
	public boolean isSatisfying(FeaturizedDataSet<D, L> data, D datum) {
		return this.firstConstraint.isSatisfying(data, datum) 
				&& this.secondConstraint.isSatisfying(data, datum);
	}
}
