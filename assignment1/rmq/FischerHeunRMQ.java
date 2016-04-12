package rmq;
import java.util.ArrayDeque;
import java.util.Deque;
/**
 * An &lt;O(n), O(1)&gt; implementation of the Fischer-Heun RMQ data structure.
 *
 * You will implement this class for problem 3.iv of Problem Set One.
 */
public class FischerHeunRMQ implements RMQ {
	private final int blockSize;
	private final int[][] topLevelSparseTable;
	private final int[] topLevel; // The array of index of minimum entry in the top level
	private final int[] logTable; // logTable[i] contains [log2(i)]
	private final float[] arr;
	private final int[][][] cache;
	private final int[] cartesianTreeNumberTable; // The array containing the cartesian tree number

	/**
	 * Creates a new FischerHeunRMQ structure to answer queries about the
	 * array given by elems.
	 *
	 * @elems The array over which RMQ should be computed.
	 */
	public FischerHeunRMQ(float[] elems) {
		// TODO: Implement this!
		int n = elems.length;
		blockSize = (int) (0.25 * Math.log(n + 1) / Math.log(2)) + 1;
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
		/**
		 * Preprocess the bottom level full precomputation data structure
		 * Make a table of length 4^b storing pointers to RMQ structures.
		 * With index corresponds to the Cartesian tree number.
		 * When computing the RMQ for a particular block, first compute its Cartesian tree number t.
		 * If there's an RMQ structure for t in the array, use it.
		 * Otherwise, compute the RMQ structure for the current block, store it in the array and index t, then use it.
		 */
		/**
		 * We use a three-dimensional array to keep track of the RMQ structures
		 * cache[i] contains the full preprocessing table of i-th block
		 */
		cache = new int[1 << (blockSize * 2)][blockSize][blockSize];
		// Initialize to -1
		for (int i = 0; i < (1 << (blockSize * 2)); i++) {
			for (int j = 0; j < blockSize; j++) {
				for (int l = 0; l < blockSize; l++) {
					cache[i][j][l] = -1;
				}
			}
		}
		// For each block, compute the Cartesian tree number
		// Check if the cartesian tree is used
		// Build an array indicating the cartesian tree number of each block by blockIdx
		cartesianTreeNumberTable = new int[blockNum];
		for (int blockIdx = 0; blockIdx < blockNum; blockIdx++) {
			int currCartesianTreeNumber = cartesianTreeNumber(blockIdx * blockSize);
			if (cache[currCartesianTreeNumber][0][0] == -1) {
				// Unassigned, need to assign the true values
				// Use dynamic programming to fill in the index of minimum entry within ranges
				for (int len = 0; len < blockSize; len++) {
					for (int i = 0; i + len < blockSize && blockIdx * blockSize + i < n; i++) {
						if (len == 0) {
							cache[currCartesianTreeNumber][i][i + len] = i;
						} else {
							int candidate1 = cache[currCartesianTreeNumber][i][i + len - 1];
							int candidate2 = cache[currCartesianTreeNumber][i + 1][i + len];
							cache[currCartesianTreeNumber][i][i + len] = elems[blockIdx * blockSize + candidate1] < elems[blockIdx * blockSize + candidate2] ? candidate1 : candidate2;
						}
					}
				}
			}
			cartesianTreeNumberTable[blockIdx] = currCartesianTreeNumber;
		}
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
			if (i < beginBlockIdx * blockSize) {
				// Query in (beginBlockIdx - 1)-th block
				int candidate = cache[cartesianTreeNumberTable[beginBlockIdx - 1]][i - (beginBlockIdx - 1) * blockSize][blockSize - 1] + (beginBlockIdx - 1) * blockSize;
				if (arr[candidate] < arr[rtn]) {
					rtn = candidate;
				}
			}
			if (j >= (endBlockIdx + 1) * blockSize) {
				// Query in (endBlockIdx + 1)-th block
				int candidate = cache[cartesianTreeNumberTable[endBlockIdx + 1]][0][j - (endBlockIdx + 1) * blockSize] + (endBlockIdx + 1) * blockSize;
				if (arr[candidate] < arr[rtn]) {
					rtn = candidate;
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

	/**
	 * Helper function to derive the Cartesian tree number of an array
	 * Input:
	 * startIdx - the starting index
	 * Output:
	 * The Cartesian tree number of arr[startIdx, startIdx + blockSize - 1]
	 * Pad with zeros to make the sequence length 2 * blockSize
	 * Runtime is O(blockSize)
	 */
	public int cartesianTreeNumber(int startIdx) {
		// The cartesian tree number: push - 2 * rtn + 1; pop - 2 * rtn
		int rtn = 0;
		Deque<Float> st = new ArrayDeque<Float>(); // The stack for the right spine
		int seqLength = 0;
		int n = arr.length;
		for (int idx = startIdx; idx < startIdx + blockSize && idx < n; idx++) {
			// Pop the stack until it's empty or the top node has a lower value than the current value.
			float curr = arr[idx];
			while (!st.isEmpty() && st.peekFirst() >= curr) {
				st.removeFirst();
				rtn *= 2;
				seqLength++;
			}
			// Push the new node onto the stack
			st.addFirst(new Float(curr));
			rtn = rtn * 2 + 1;
			seqLength++;
		}
		// Padding with 0s to make seqLength = blockSize * 2
		while (seqLength < blockSize * 2) {
			rtn *= 2;
			seqLength++;
		}
		return rtn;
	}
}
