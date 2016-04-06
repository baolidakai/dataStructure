#!/usr/bin/python3
'''
Implements the Aho-Corasick automaton
'''
from collections import deque
patterns = list(map(lambda x: x.strip(), open('patterns.txt', 'r')))
letters = [chr(ord('a') + i) for i in range(26)]

class trieNode:
	'''
	Trie Node data structure
	Each trie node contains a boolean indicating whether it is a leaf
	An array of size 26 representing the English alphabet
	A pointer to suffix link, None if not any
	A pointer to output link, None if not any
	'''
	def __init__(self, parent = None, level = 0, isLeaf = False):
		self.leaf = isLeaf
		self.neighbors = [None] * 26
		self.suffix = None
		self.output = None
		self.parent = parent
		self.level = level

	def addNeighbor(self, letter, isLeaf = False):
		# Add neighbor to the current node with letter from a-z
		self.neighbors[ord(letter) - ord('a')] = trieNode(parent = self, level = self.level + 1, isLeaf = isLeaf)

	def getNeighbor(self, path):
		# Reach the neighbor through a chain
		curr = self
		for letter in path:
			if not curr:
				return
			curr = curr.neighbors[ord(letter) - ord('a')]
		return curr

	def getNeighbors(self):
		return [(letter, self.getNeighbor(letter)) for letter in letters if self.getNeighbor(letter)]

	def addSuffixLink(self, destination):
		self.suffix = destination
	
	def addOutputLink(self, destination):
		self.output = destination
	
	def __str__(self):
		rtn = ' '.join([letter for letter in letters if self.getNeighbor(letter)])
		if self.leaf:
			rtn += ' *' # Mark for leaf
		return rtn

def constructTries(root, patterns):
	# Construct the tries from the root and the patterns
	for pattern in patterns:
		patternLength = len(pattern)
		curr = root
		for i, char in enumerate(pattern):
			if not curr.getNeighbor(char):
				curr.addNeighbor(char, i == patternLength - 1)
			curr = curr.getNeighbor(char)

def addLinks(root):
	'''
	Construct the suffix links and output links
	For suffix links:
		Do a breadth-first search of the trie
		If the node is the root, no suffix link
		If the node is one hop away from the root, points to the root
		Otherwise, the node corresponds to some string wa
		Let w->x:
			If xa exists, wa->xa
			Else if x is the root node, wa->root
			Else x->x.suffix
	For output links:
		u = v.suffix
		If u is a leaf, set v.output = u
		otherwise, v.output = u.output
	'''
	# The elements in the queue contains the node itself, the level, and the letter corresponding to the current node
	frontier = deque([(root, 0, None)])
	while frontier:
		node, level, letter = frontier.popleft()
		# Add the suffix link
		if level == 1: # node is one hop away
			node.addSuffixLink(root)
		elif level != 0:
			x = node.parent.suffix
			while True:
				if x.getNeighbor(letter):
					node.addSuffixLink(x.getNeighbor(letter))
					break
				elif x == root:
					node.addSuffixLink(root)
					break
				else:
					x = x.suffix
		# Add the output link
		if level != 0:
			u = node.suffix
			if u.leaf:
				node.output = u
			else:
				node.output = u.output
		# Add all neighbors
		for letter, neighbor in node.getNeighbors():
			frontier.append((neighbor, level + 1, letter))

class automaton:
	'''
	automaton class preprocessing the patterns to construct the automaton with links
	and query function to search for all occurrences
	'''
	def __init__(self, patterns):
		self.root = trieNode()
		constructTries(self.root, patterns)
		addLinks(self.root)

root = trieNode()
constructTries(root, patterns)
addLinks(root)

text = list(open('text.txt', 'r'))[0].strip()
'''
The final matching algorithm
Start at the root node in the trie
For each character c in the string:
	while no edge labeled c:
		if at the root, break; otherwise, follow a suffix link
	if there is an edge labeled c, follow it
	If the current node is a leaf, output that pattern
	Output all the words in the chain of output links originating at this node.
'''
print('Automaton...')
curr = root
automatonOutputs = set()
for i, c in enumerate(text):
	while not curr.getNeighbor(c):
		if curr == root:
			break
		curr = curr.suffix
	if curr.getNeighbor(c):
		curr = curr.getNeighbor(c)
	if curr.leaf:
		automatonOutputs.add((i - curr.level + 1, i))
	outputNode = curr
	while outputNode.output:
		automatonOutputs.add((i - outputNode.output.level + 1, i))
		outputNode = outputNode.output

print('Brute force search...')
# Compare the results from the naive brute force search algorithm
bruteforceOutputs = set()
for i in range(len(text)):
	for j in range(i, len(text)):
		if text[i:j + 1] in patterns:
			bruteforceOutputs.add((i, j))

if automatonOutputs != bruteforceOutputs:
	raise Exception('Incorrect match result!')
else:
	print('Correct matching!')
