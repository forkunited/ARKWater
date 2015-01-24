package ark.util;

import ark.data.annotation.Datum;

public interface Parameterizable<D extends Datum<L>, L> {
	/**
	 * @return parameters that can be set
	 */
	String[] getParameterNames();
	
	/**
	 * @param parameter
	 * @return the value of the given parameter
	 */
	String getParameterValue(String parameter);
	
	/**
	 * 
	 * @param parameter
	 * @param parameterValue
	 * @param datumTools
	 * @return true if the parameter has been set to parameterValue.  Some parameters are set to
	 * objects retrieved through datumTools that are named by parameterValue.
	 */
	boolean setParameterValue(String parameter, String parameterValue, Datum.Tools<D, L> datumTools);
}
