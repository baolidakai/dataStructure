package rmq;
/**
 * An &lt;O(n), O(log n)&gt; implementation of the RMQ as a hybrid between
 * the sparse table (on top) and no-precomputation structure (on bottom)
 *
 * You will implement this class for problem 3.iii of Problem Set One.
 */
public class HybridRMQ implements RMQ {
	private final int blockSize;
	private final int[][] topLevelSparseTable;
	private final int[] topLevel; // The array of index of minimum entry in the top level
	private final int[] logTable; // logTable[i] contains [log2(i)]
	private final float[] arr;

	/**
	 * Creates a new HybridRMQ structure to answer queries about the
	 * array given by elems.
	 *
	 * @elems The array over which RMQ should be computed.
	 */
	public HybridRMQ(float[] elems) {
		// TODO: Implement this!
		/**
		 * Set the block size to log n
		 * Use a sparse table for the top-level structure.
		 * Use the "no preprocessing" structure for each block.
		 */
		int n = elems.length;
		blockSize = (int) (Math.log(n) / Math.log(2)) + 1;
		int blockNum = (int) Math.ceil((double) n / blockSize);
		// Compute the index of minimum entry in top level
		topLevel = new int[blockNum];
		// Initialize to -1
		for (int i = 0; i < blockNum; i++) {
			topLevel[i] = -1;
		}
		// Update the minimum entries when necessary
		for (int i = 0; i < n; i++) {
			int currBlockIdx = i / blockSize;
			int currBlockMinIdx = topLevel[currBlockIdx];
			if (currBlockMinIdx == -1 || elems[i] < elems[currBlockMinIdx]) {
				topLevel[currBlockIdx] = i;
			}
		}
		// Create the lookup table for [log2(n)]
		logTable = new int[blockNum + 1];
		logTable[0] = -1;
		for (int i = 1; i <= blockNum; i++) {
			logTable[i] = logTable[i >> 1] + 1;
		}
		int k = logTable[blockNum];
		topLevelSparseTable = new int[blockNum][k + 1];
		// Fill column by column
		// sparseTable[i][j] = argmin(sparseTable[i][j - 1], sparseTable[i + 2 ^ (j - 1)][j - 1])
		for (int j = 0; j <= k; j++) {
			for (int i = 0; i + (1 << j) <= blockNum; i++) {
				if (j == 0) {
					topLevelSparseTable[i][j] = topLevel[i];
				} else {
					int candidate1 = topLevelSparseTable[i][j - 1];
					int candidate2 = topLevelSparseTable[i + (1 << (j - 1))][j - 1];
					topLevelSparseTable[i][j] = elems[candidate1] < elems[candidate2] ? candidate1 : candidate2;
				}
			}
		}
		arr = elems;
	}

	/**
	 * Evaluates RMQ(i, j) over the array stored by the constructor, returning
	 * the index of the minimum value in that range.
	 */
	@Override
	public int rmq(int i, int j) {
		// TODO: Implement this!
		// Find the minimum index in the top level
		int beginBlockIdx = (int) Math.ceil((double) i / blockSize);
		int endBlockIdx = (int) Math.floor((double) (j + 1) / blockSize) - 1;
		int rtn = -1; // Result
		if (beginBlockIdx <= endBlockIdx) {
			int k = logTable[endBlockIdx - beginBlockIdx + 1];
			int candidate1 = topLevelSparseTable[beginBlockIdx][k];
			int candidate2 = topLevelSparseTable[endBlockIdx - (1 << k) + 1][k];
			rtn = arr[candidate1] < arr[candidate2] ? candidate1 : candidate2;
			// Find the minimum index in the bottom level
			for (int idx = i; idx < beginBlockIdx * blockSize; idx++) {
				if (rtn == -1 || arr[idx] < arr[rtn]) {
					rtn = idx;
				}
			}
			for (int idx = j; idx >= (endBlockIdx + 1) * blockSize; idx--) {
				if (rtn == -1 || arr[idx] < arr[rtn]) {
					rtn = idx;
				}
			}
		} else {
			// Suffices to search within the current [i, j]
			for (int idx = i; idx <= j; idx++) {
				if (rtn == -1 || arr[idx] < arr[rtn]) {
					rtn = idx;
				}
			}
		}
		return rtn;
	}
}
