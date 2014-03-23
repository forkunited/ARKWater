package ark.model;

import java.io.Reader;
import java.io.Writer;
import java.util.Map;

import ark.data.annotation.Datum;
import ark.data.annotation.Datum.Tools;
import ark.data.feature.FeaturizedDataSet;

public class SupervisedModelCreg<D extends Datum<L>, L> extends SupervisedModel<D, L> {

	@Override
	protected String[] getHyperParameterNames() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected SupervisedModel<D, L> makeInstance() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected boolean deserializeExtraInfo(Reader reader, Tools<D, L> datumTools) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	protected boolean deserializeParameters(Reader reader, Tools<D, L> datumTools) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	protected boolean serializeParameters(Writer writer) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	protected boolean serializeExtraInfo(Writer writer) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public String getGenericName() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getHyperParameterValue(String parameter) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean setHyperParameterValue(String parameter,
			String parameterValue, Tools<D, L> datumTools) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public boolean train(FeaturizedDataSet<D, L> data) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public Map<D, Map<L, Double>> posterior(FeaturizedDataSet<D, L> data) {
		// TODO Auto-generated method stub
		return null;
	}

}
