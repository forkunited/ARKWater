package ark.model.evaluation;

public class ValidationEMGST {
	public ValidationEMGST() {
		// Input: ValidationGST
		// Train Labeled data
		// Dev Labeled data
		// Test Labeled data
		// Unlabeled train data
		// Unlabeled test data

		// Iteration 0:
		// Train on labeled data, evaluate on test labeled data
		// Get the model out of GST validation and use it on unlabeled train+test data
		// Compute unlabeled evaluations on combined labeled + unlabeled test data
		
		// Iteration 1...n:
		// Train on labeled+unlabeled train data, evaluate on labeled test data
		// Compute unlabeled evaluations on combined labeled + unlabeled test data
		
		
		
	}
}
