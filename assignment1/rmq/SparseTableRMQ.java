package rmq;
/**
 * An &lt;O(n log n), O(1)&gt; implementation of RMQ that uses a sparse table
 * to do lookups efficiently.
 *
 * You will implement this class for problem 3.ii of Problem Set One.
 */
public class SparseTableRMQ implements RMQ {
	private final int[][] sparseTable;
	private final float[] arr;
	private final int[] logTable; // logTable[i] contains [log2(i)]

	/**
	 * Creates a new SparseTableRMQ structure to answer queries about the
	 * array given by elems.
	 *
	 * @elems The array over which RMQ should be computed.
	 */
	public SparseTableRMQ(float[] elems) {
		// TODO: Implement this!
		/**
		 * Use a sparse table to store the index of ranges:
		 * sparseTable[i][k] contains the index of minimum entry in range [i, i + 2^k - 1]
		 * Also need a preprocess that computes [log2(j - i + 1)] in O(1) time
		 */
		// Lookup table for [log2(*)]
		int n = elems.length;
		logTable = new int[n + 1];
		// We use dynamic programming to fill in this log table
		// Base case: logTable[1] = 0
		// Recurrence: logTable[i] = logTable[i / 2] + 1
		logTable[0] = -1;
		for (int i = 1; i <= n; i++) {
			logTable[i] = logTable[i >> 1] + 1;
		}
		int k = logTable[n];
		sparseTable = new int[n][k + 1];
		// Fill column by column
		// sparseTable[i][j] = argmin(sparseTable[i][j - 1], sparseTable[i + 2 ^ (j - 1)][j - 1])
		for (int j = 0; j <= k; j++) {
			for (int i = 0; i + (1 << j) <= n; i++) {
				if (j == 0) {
					sparseTable[i][j] = i;
				} else {
					int candidate1 = sparseTable[i][j - 1];
					int candidate2 = sparseTable[i + (1 << (j - 1))][j - 1];
					sparseTable[i][j] = elems[candidate1] < elems[candidate2] ? candidate1 : candidate2;
				}
			}
		}
		arr = new float[n];
		for (int i = 0; i < n; i++) {
			arr[i] = elems[i];
		}
	}

	/**
	 * Evaluates RMQ(i, j) over the array stored by the constructor, returning
	 * the index of the minimum value in that range.
	 */
	@Override
	public int rmq(int i, int j) {
		// TODO: Implement this!
		// Compute the largest k such that i + 2^k - 1 <= j, i.e. 2^k <= j - i + 1
		int k = logTable[j - i + 1];
		// Return the argmin of sparseTable[i][k] and sparseTable[j - 2 ^ k + 1][k]
		int candidate1 = sparseTable[i][k];
		int candidate2 = sparseTable[j - (1 << k) + 1][k];
		int rtn = arr[candidate1] < arr[candidate2] ? candidate1 : candidate2;
		return rtn;
	}
}
