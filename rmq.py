'''
Range Minimum Queries implementations
'''
import math
import random
import time
import timeit
import matplotlib.pyplot as plt
import numpy as np
# First implementation:
# No preprocessing
# There is no preprocessing time
# For each query, linearly scan from the two indices and find the minimum in that range inclusively
class noPreprocessing:
	def __init__(self, arr):
		self.arr = arr
		self.size = len(arr)
	def minimumInRange(self, left, right):
		if left < 0 or right >= self.size or left > right:
			raise Exception('Indices out of boundary')
		rtn = float('Inf')
		for idx in range(left, right + 1):
			if self.arr[idx] < rtn:
				rtn = self.arr[idx]
		return rtn

# Second implementation:
# Full preprocessing
# Constant query time
'''
Helper data structure:
triangular array
Supported operations:
Assignment
Random Access
'''
class triangularArray:
	def __init__(self, size):
		self.arr = [None] * (size * (size + 1) // 2)
		self.size = size
	def setAt(self, row, col, value):
		if row < 0 or col >= self.size or row > col:
			raise Exception('Assignment out of bound!')
		idx = self.size * row + col - row - (row - 1) * row // 2
		self.arr[idx] = value
	def getAt(self, row, col):
		if row < 0 or col >= self.size or row > col:
			raise Exception('Access out of bound!')
		idx = self.size * row + col - row - (row - 1) * row // 2
		return self.arr[idx]

class fullPreprocessing:
	def __init__(self, arr):
		# Compute the triangular array for the range queries
		self.size = len(arr)
		self.cache = triangularArray(self.size)
		for length in range(1, self.size + 1):
			for row in range(self.size - length + 1):
				if length == 1:
					self.cache.setAt(row, row + length - 1, arr[row])
				else:
					currMin = min(self.cache.getAt(row, row + length - 2), self.cache.getAt(row + 1, row + length - 1))
					self.cache.setAt(row, row + length - 1, currMin)
	def minimumInRange(self, left, right):
		if left < 0 or right >= self.size or left > right:
			raise Exception('Indices out of boundary')
		return self.cache.getAt(left, right)

# Third implementation
# Block decomposition
# Decompose the array into sqrt(N) blocks, each of sqrt(N) consecutive elements
# Pre-compute the minimum in each block with brute force
# For query, linearly scan the outliers and the blocks within the range
class blockDecomposition:
	def __init__(self, arr):
		self.size = len(arr)
		self.blockSize = int(math.sqrt(self.size)) + 1
		self.blockNum = int(math.sqrt(self.size)) + 1
		self.arr = arr
		# Pre-compute the minimum in each block
		self.blockMin = [float('Inf')] * self.blockNum
		for idx in range(self.size):
			blockIdx = idx // self.blockSize
			if arr[idx] < self.blockMin[blockIdx]:
				self.blockMin[blockIdx] = arr[idx]
	def minimumInRange(self, left, right):
		if left < 0 or right >= self.size or left > right:
			raise Exception('Indices out of boundary')
		rtn = float('Inf')
		# Scan the outliers - i.e. bottom elements
		# Left outliers
		leftLower = left
		leftUpper = min(right, math.ceil(left / self.blockSize) * self.blockSize)
		if leftUpper % self.blockSize == 0:
			leftUpper -= 1
		for leftPtr in range(leftLower, leftUpper + 1):
			rtn = min(rtn, self.arr[leftPtr])
		# Right outliers
		rightUpper = right
		rightLower = max(left, right // self.blockSize * self.blockSize)
		for rightPtr in range(rightLower, rightUpper + 1):
			rtn = min(rtn, self.arr[rightPtr])
		# Scan the minimum of blocks in the range
		topLevelLeft = math.ceil(left / self.blockSize)
		topLevelRight = (right + 1) // self.blockSize - 1
		for blockIdx in range(topLevelLeft, topLevelRight + 1):
			rtn = min(rtn, self.blockMin[blockIdx])
		return rtn

# Fourth implementation
# Sparse table implementation
# Preprocessing: using dynamic programming to store the minimum of range [i, i + 2^k) for any valid k
# Query: Access at most 2 results stored in the sparse table, and compute their minimum
class sparseTable:
	def __init__(self, arr):
		'''
		Construct the sparse table with dynamic programming
		dp[i][k] contains the minimum of range [i, i + 2^k]
		Recursion: dp[i][k] = min(dp[i][k - 1], dp[i + 2^(k - 1)][k - 1])
		'''
		self.size = len(arr)
		self.dp = [[None for _ in range(int(math.log2(self.size)) + 1)] for _ in range(self.size)]
		for k in range(int(math.log2(self.size)) + 1):
			for i in range(self.size - 2 ** k + 1):
				if k == 0:
					self.dp[i][k] = arr[i]
				else:
					self.dp[i][k] = min(self.dp[i][k - 1], self.dp[i + 2 ** (k - 1)][k - 1])
	def minimumInRange(self, left, right):
		'''
		Return the minimum in range [left, right] inclusive
		access self.dp to ask for at most two necessary information
		'''
		# Compute the maximum possible k within range
		k = int(math.log2(right - left + 1))
		return min(self.dp[left][k], self.dp[right - 2 ** k + 1][k])
	def __str__(self):
		return ' '.join(map(str, self.dp))

# Hybrid 1
# Set the block size to log n
# Use a sparse table for the top-level structure
# Use the ``no preprocessing'' structure for each block.
# The major difference between this data structure and block decomposition
# is that when querying the top-level, we use the sparse table instead
class hybridOne:
	def __init__(self, arr):
		self.size = len(arr)
		self.blockSize = int(math.log(self.size)) + 1
		self.blockNum = self.size // self.blockSize + 1
		self.arr = arr
		# Pre-compute the minimum in each block
		self.blockMin = [float('Inf')] * self.blockNum
		for idx in range(self.size):
			blockIdx = idx // self.blockSize
			if arr[idx] < self.blockMin[blockIdx]:
				self.blockMin[blockIdx] = arr[idx]
		# Construct sparse table for the blockMin array
		self.st = sparseTable(self.blockMin)
	def minimumInRange(self, left, right):
		if left < 0 or right >= self.size or left > right:
			raise Exception('Indices out of boundary')
		rtn = float('Inf')
		# Scan the outliers - i.e. bottom elements
		# Left outliers
		leftLower = left
		leftUpper = min(right, math.ceil(left / self.blockSize) * self.blockSize)
		if leftUpper % self.blockSize == 0:
			leftUpper -= 1
		for leftPtr in range(leftLower, leftUpper + 1):
			rtn = min(rtn, self.arr[leftPtr])
		# Right outliers
		rightUpper = right
		rightLower = max(left, right // self.blockSize * self.blockSize)
		for rightPtr in range(rightLower, rightUpper + 1):
			rtn = min(rtn, self.arr[rightPtr])
		# Scan the minimum of blocks in the range
		topLevelLeft = math.ceil(left / self.blockSize)
		topLevelRight = (right + 1) // self.blockSize - 1
		if topLevelLeft <= topLevelRight:
			rtn = min(rtn, self.st.minimumInRange(topLevelLeft, topLevelRight))
		return rtn

# Hybrid 2
# Use the sparse table for both the top and bottom RMQ structures with a block size of log n
# The major difference between this data structure and hybrid 1 is that we also use sparse table
# for bottom structure
class hybridTwo:
	def __init__(self, arr):
		self.size = len(arr)
		self.blockSize = int(math.log(self.size)) + 1
		self.blockNum = math.ceil(self.size / self.blockSize)
		self.arr = arr
		# Pre-compute the minimum in each block
		self.blockMin = [float('Inf')] * self.blockNum
		for idx in range(self.size):
			blockIdx = idx // self.blockSize
			if arr[idx] < self.blockMin[blockIdx]:
				self.blockMin[blockIdx] = arr[idx]
		# Construct sparse table for the blockMin array
		self.topSparseTable = sparseTable(self.blockMin)
		# Construct sparse table for every block
		self.bottomSparseTables = [None] * self.blockNum
		for blockIdx in range(self.blockNum):
			# Construct sparse table for current block
			self.bottomSparseTables[blockIdx] = sparseTable(arr[blockIdx * self.blockSize:(blockIdx + 1) * self.blockSize])
	def minimumInRange(self, left, right):
		if left < 0 or right >= self.size or left > right:
			raise Exception('Indices out of boundary')
		rtn = float('Inf')
		# Scan the outliers - i.e. bottom elements
		# Left outliers
		leftLower = left
		leftUpper = min(right, math.ceil(left / self.blockSize) * self.blockSize)
		leftBlockIdx = leftLower // self.blockSize
		if leftUpper - leftBlockIdx * self.blockSize == self.blockSize:
			leftUpper -= 1
		rtn = min(rtn, self.bottomSparseTables[leftBlockIdx].minimumInRange(leftLower - leftBlockIdx * self.blockSize, leftUpper - leftBlockIdx * self.blockSize))
		for leftPtr in range(leftLower, leftUpper + 1):
			rtn = min(rtn, self.arr[leftPtr])
		# Right outliers
		rightUpper = right
		rightLower = max(left, right // self.blockSize * self.blockSize)
		for rightPtr in range(rightLower, rightUpper + 1):
			rtn = min(rtn, self.arr[rightPtr])
		# Scan the minimum of blocks in the range
		topLevelLeft = math.ceil(left / self.blockSize)
		topLevelRight = (right + 1) // self.blockSize - 1
		if topLevelLeft <= topLevelRight:
			rtn = min(rtn, self.topSparseTable.minimumInRange(topLevelLeft, topLevelRight))
		return rtn

# Hybrid 3
# Use sparse table for the top structure and hybrid one for the bottom structure
class hybridThree:
	def __init__(self, arr):
		self.size = len(arr)
		self.blockSize = int(math.log(self.size)) + 1
		self.blockNum = math.ceil(self.size / self.blockSize)
		self.arr = arr
		# Pre-compute the minimum in each block
		self.blockMin = [float('Inf')] * self.blockNum
		for idx in range(self.size):
			blockIdx = idx // self.blockSize
			if arr[idx] < self.blockMin[blockIdx]:
				self.blockMin[blockIdx] = arr[idx]
		# Construct sparse table for the blockMin array
		self.topSparseTable = sparseTable(self.blockMin)
		# Construct sparse table for every block
		self.bottomHybrids = [None] * self.blockNum
		for blockIdx in range(self.blockNum):
			# Construct sparse table for current block
			self.bottomHybrids[blockIdx] = hybridOne(arr[blockIdx * self.blockSize:(blockIdx + 1) * self.blockSize])
	def minimumInRange(self, left, right):
		if left < 0 or right >= self.size or left > right:
			raise Exception('Indices out of boundary')
		rtn = float('Inf')
		# Scan the outliers - i.e. bottom elements
		# Left outliers
		leftLower = left
		leftUpper = min(right, math.ceil(left / self.blockSize) * self.blockSize)
		leftBlockIdx = leftLower // self.blockSize
		if leftUpper - leftBlockIdx * self.blockSize == self.blockSize:
			leftUpper -= 1
		rtn = min(rtn, self.bottomHybrids[leftBlockIdx].minimumInRange(leftLower - leftBlockIdx * self.blockSize, leftUpper - leftBlockIdx * self.blockSize))
		for leftPtr in range(leftLower, leftUpper + 1):
			rtn = min(rtn, self.arr[leftPtr])
		# Right outliers
		rightUpper = right
		rightLower = max(left, right // self.blockSize * self.blockSize)
		for rightPtr in range(rightLower, rightUpper + 1):
			rtn = min(rtn, self.arr[rightPtr])
		# Scan the minimum of blocks in the range
		topLevelLeft = math.ceil(left / self.blockSize)
		topLevelRight = (right + 1) // self.blockSize - 1
		if topLevelLeft <= topLevelRight:
			rtn = min(rtn, self.topSparseTable.minimumInRange(topLevelLeft, topLevelRight))
		return rtn

def checkCorrectness(n = 100, m = 100, minNum = 0, maxNum = 20):
	arr = [random.randint(minNum, maxNum) for _ in range(n)]
	objects = [noPreprocessing(arr), fullPreprocessing(arr), blockDecomposition(arr), sparseTable(arr), hybridOne(arr), hybridTwo(arr), hybridThree(arr)]
	for _ in range(m):
		left = random.randint(0, n - 1)
		right = random.randint(left, n - 1)
		results = list(map(lambda obj: obj.minimumInRange(left, right), objects))
		for result in results:
			if result != results[0]:
				print('Inconsistent queries detected with range [{0}, {1}]!'.format(left, right))
				print(results)
				return False
	return True

#Visualize the amortized preprocessing time and query time
maxN = 10000
numExperiment = 10
minNum = 0
maxNum = 20
Ns = list(map(lambda k: 2 ** k, range(int(math.log2(maxN)))))
npPreprocessTimes = np.zeros_like(Ns)
fpPreprocessTimes = np.zeros_like(Ns)
bdPreprocessTimes = np.zeros_like(Ns)
stPreprocessTimes = np.zeros_like(Ns)
hoPreprocessTimes = np.zeros_like(Ns)
htPreprocessTimes = np.zeros_like(Ns)
hhPreprocessTimes = np.zeros_like(Ns)
npQueryTimes = np.zeros_like(Ns)
fpQueryTimes = np.zeros_like(Ns)
bdQueryTimes = np.zeros_like(Ns)
stQueryTimes = np.zeros_like(Ns)
hoQueryTimes = np.zeros_like(Ns)
htQueryTimes = np.zeros_like(Ns)
hhQueryTimes = np.zeros_like(Ns)
for i, N in enumerate(Ns):
	arr = [random.randint(minNum, maxNum) for _ in range(N)]
	# Compute the runtimes for no preprocessing
	timeStart = time.time()
	npObject = noPreprocessing(arr)
	npPreprocessTimes[i] = (time.time() - timeStart) * 1000000
	timeStart = time.time(); fpObject = fullPreprocessing(arr); fpPreprocessTimes[i] = (time.time() - timeStart) * 1000000
	timeStart = time.time(); bdObject = blockDecomposition(arr); bdPreprocessTimes[i] = (time.time() - timeStart) * 1000000
	timeStart = time.time(); stObject = sparseTable(arr); stPreprocessTimes[i] = (time.time() - timeStart) * 1000000
	timeStart = time.time(); hoObject = hybridOne(arr); hoPreprocessTimes[i] = (time.time() - timeStart) * 1000000
	timeStart = time.time(); htObject = hybridTwo(arr); htPreprocessTimes[i] = (time.time() - timeStart) * 1000000
	timeStart = time.time(); hhObject = hybridThree(arr); hhPreprocessTimes[i] = (time.time() - timeStart) * 1000000
	for experiment in range(numExperiment):
		left = random.randint(0, N - 1)
		right = random.randint(0, N - 1)
		left, right = min(left, right), max(left, right)
		timeStart = time.time(); _ = npObject.minimumInRange(left, right); npQueryTimes[i] += (time.time() - timeStart) * 1000000
		timeStart = time.time(); _ = fpObject.minimumInRange(left, right); fpQueryTimes[i] += (time.time() - timeStart) * 1000000
		timeStart = time.time(); _ = bdObject.minimumInRange(left, right); bdQueryTimes[i] += (time.time() - timeStart) * 1000000
		timeStart = time.time(); _ = stObject.minimumInRange(left, right); stQueryTimes[i] += (time.time() - timeStart) * 1000000
		timeStart = time.time(); _ = hoObject.minimumInRange(left, right); hoQueryTimes[i] += (time.time() - timeStart) * 1000000
		timeStart = time.time(); _ = htObject.minimumInRange(left, right); htQueryTimes[i] += (time.time() - timeStart) * 1000000
		timeStart = time.time(); _ = hhObject.minimumInRange(left, right); hhQueryTimes[i] += (time.time() - timeStart) * 1000000
	npQueryTimes[i] /= numExperiment
	fpQueryTimes[i] /= numExperiment
	bdQueryTimes[i] /= numExperiment
	stQueryTimes[i] /= numExperiment
	hoQueryTimes[i] /= numExperiment
	htQueryTimes[i] /= numExperiment
	hhQueryTimes[i] /= numExperiment
ax1 = plt.subplot(211)
ax1.plot(Ns, npPreprocessTimes, label = 'No Preprocessing')
ax1.plot(Ns, bdPreprocessTimes, label = 'Block Decomposition')
ax1.plot(Ns, hoPreprocessTimes, label = 'Hybrid One')
ax1.plot(Ns, hhPreprocessTimes, label = 'Hybrid Three')
ax1.plot(Ns, htPreprocessTimes, label = 'Hybrid Two')
ax1.plot(Ns, stPreprocessTimes, label = 'Sparse Table')
ax1.plot(Ns, fpPreprocessTimes, label = 'Full Preprocessing')
ax1.set_ylim([0, 2000])
ax1.set_xlabel('N')
ax1.set_ylabel('Preprocess time(ms/1000)')
ax1.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.subplots_adjust(right = 0.7)
ax2 = plt.subplot(212)
ax2.plot(Ns, npQueryTimes, label = 'No Preprocessing')
ax2.plot(Ns, bdQueryTimes, label = 'Block Decomposition')
ax2.plot(Ns, hoQueryTimes, label = 'Hybrid One')
ax2.plot(Ns, hhQueryTimes, label = 'Hybrid Three')
ax2.plot(Ns, htQueryTimes, label = 'Hybrid Two')
ax2.plot(Ns, stQueryTimes, label = 'Sparse Table')
ax2.plot(Ns, fpQueryTimes, label = 'Full Preprocessing')
ax2.set_ylim([0, 10])
ax2.set_xlabel('N')
ax2.set_ylabel('Amortized Query time(ms/1000)')
ax2.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.subplots_adjust(right = 0.7)
plt.savefig('rangeMinimumQuery')
