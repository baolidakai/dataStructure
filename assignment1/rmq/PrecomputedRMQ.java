package rmq;
/**
 * An &lt;O(n<sup>2</sup>), O(1)&gt; implementation of RMQ that precomputes the
 * value of RMQ_A(i, j) for all possible i and j.
 *
 * You will implement this class for problem 3.i of Problem Set One.
 */
public class PrecomputedRMQ implements RMQ {
	private int[][] idxLocation; // idxLocation[i][j] stores the index of minimum element in range [i, j]
	
	/**
	 * Creates a new PrecomputedRMQ structure to answer queries about the
	 * array given by elems.
	 *
	 * @elems The array over which RMQ should be computed.
	 */
	public PrecomputedRMQ(float[] elems) {
		// TODO: Implement this! 
		int n = elems.length;
		idxLocation = new int[n][n];
		// For i <= j, fill in idxLocation[i][j] with the index of the minimum entry in the range
		// We use dynamic programming to implement this
		// Recurse on length of interval:
		// idxLocation[i][i + L] = argmin(idxLocation[i][i + L - 1], idxLocation[i + 1][i + L])
		for (int L = 0; L < n; L++) {
			for (int i = 0; i + L < n; i++) {
				if (L == 0) {
					idxLocation[i][i + L] = i;
				} else {
					int candidate1 = idxLocation[i][i + L - 1];
					int candidate2 = idxLocation[i + 1][i + L];
					idxLocation[i][i + L] = elems[candidate1] < elems[candidate2] ? candidate1 : candidate2;
				}
			}
		}
	}

	/**
	 * Evaluates RMQ(i, j) over the array stored by the constructor, returning
	 * the index of the minimum value in that range.
	 */
	@Override
	public int rmq(int i, int j) {
		// TODO: Implement this!
		int rtn = idxLocation[i][j];
		return rtn;
	}
}
