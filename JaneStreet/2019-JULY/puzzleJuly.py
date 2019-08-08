# Author: Hayk Aleksanyan
# JaneStreet puzzle for JULY 2019, see https://www.janestreet.com/puzzles/scroggle/

# for scraggle, one needs to use a dictionary provided at
# http://scrabble.merriam.com/ following the Terms of Use described there
# for demonstration purposes, we will use a standard english dictionary instead

import numpy as np

def readDictionary_FromText():
    """read the dictionary of words to be used in the game"""
    dd = []
    with open('english_2-15.txt') as f:
        dd = f.readlines()

    i = 0
    while i < len(dd):
        if len(dd[i]) == 0 or dd[i] == '\n':
            del dd[i]
        else:
            dd[i] = dd[i][:-1]  # drop the new line char at the end
            i += 1

    return dd


def removeNonDistinct_Consonants(words, chars):
    """ remove all words which have non-distinct consonants """
    res = []
    for word in words:
        X = dict()
        for c in word:
            if not c in X:
                X[c] = 0
            X[c] += 1

        q = False
        for c in X:
            if c in chars and X[c] > 1:
                q = True
                break

        if q == False:
            res.append(word)

    return res



class Node:
    def __init__(self, key, value = None):
        self.key = key
        self.value = value
        self.children = dict()

class Trie:
    def __init__(self, root):
        self.root = root # this is a Node

    def insert(self, word, value):
        curr_node = self.root
        for c in word:
            if not c in curr_node.children:
                curr_node.children[ c ] = Node(c)
            curr_node = curr_node.children[c]

        curr_node.value = value


    def findPrefix(self, prefix):
        curr_node = self.root
        for c in prefix:
            if not c in curr_node.children:
                return False
            curr_node = curr_node.children[c]

        return True

    def findWords_givePrefix(self, prefix):
        ans = []
        curr_node = self.root
        for c in prefix:
            if not c in curr_node.children:
                return []
            curr_node = curr_node.children[c]

        stack = [(curr_node, '')]
        while stack:
            curr, suffix = stack.pop()

            if curr.value != None:
                ans.append( (prefix + suffix, curr.value) )

            for c in curr.children:
                stack.append( (curr.children[c],  suffix +  curr.children[c].key ) )

        return ans

    def getAllWords(self):
        # compute all words stored in this tree
        ans = [ ]

        stack = [(self.root, '')]

        while stack:
            curr, s = stack.pop()
            if curr.value != None:
                ans.append( s )

            for charKey in curr.children:
                stack.append( (curr.children[charKey], s + charKey) )

        return ans


class Board:
    def __init__(self):
        self.board = Board.initBoard()
        self.chars =  list({'y', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z'})
        assert(len(self.chars) == 21)

        self.emptySlots = self.getEmptySlots()

        # this is the list of all possible char sets of 18 chars (18 = number of empty slots on the initial prefilled board)
        # by shuffling any of the choices in charChoice, we can generate a fully filled board
        self.charChoice = Board.n_choose_k( len(self.chars), 18 )

    @staticmethod
    def board_to_str(board):
        s = ""
        for i in range(6):
            s += " ".join( board[i] )
            if i < 5: s += "\n"
        return s

    def generateRandomBoard( self ):
        # return a randomly configured board based on the char choices

        k = np.random.randint(0, len( self.charChoice ))
        letter_choice = self.charChoice[k]
        np.random.shuffle(letter_choice)

        for i, (a,b) in enumerate(self.emptySlots):
            self.board[a][b] = self.chars[ letter_choice[i] ]

        return self.board

    @staticmethod
    def n_choose_k(n, k):
        # return all possible ways to choose k elements from n

        if k > n or k == 0:
            return [[]]

        if n == k:
            return [ [i for i in range(n )  ] ]

        a = Board.n_choose_k(n - 1, k - 1)
        b = Board.n_choose_k(n - 1, k)

        a = [ x + [n - 1] for x in a ]
        return a + b


    @staticmethod
    def initBoard():
        # the inital, partially filled board by JaneStreet
        b = [ 6*[''] for _ in range(6) ]

        b[0][1], b[0][3], b[0][5]  = 'o', 'e', 'u'
        b[1][0], b[1][2], b[1][4]  = 'i', 'a', 'a'
        b[2][1], b[2][3], b[2][5]  = 'e', 'i', 'o'
        b[3][0], b[3][2], b[3][4]  = 'a', 'o', 'e'
        b[4][1], b[4][3], b[4][5]  = 'e', 'a', 'i'
        b[5][0], b[5][2], b[5][4]  = 'u', 'i', 'o'

        return b

    def getEmptySlots(self):
        board_empty_slots = []

        for i in range(6):
            for j in range(6):
                if self.board[i][j] == '':
                    board_empty_slots.append( (i,j) )

        return board_empty_slots

    def generateAll_Boards( self ):
        # choose k distinct chars from @chars
        ans = []

        N = len(self.chars)
        choices = n_choose_k(N, 18 )  # 18 is the number of empty slots in the initial board

        board_empty_slots = []
        board = Board.initBoard()

        for i in range(6):
            for j in range(6):
                if board[i][j] == '':
                    board_empty_slots.append( (i,j) )

        for letter_choice in choices:
            board = initBoard()
            i = 0
            for (a,b) in board_empty_slots:
                board[a][b] = self.chars[ letter_choice[i] ]
                i += 1

            ans.append(board)

        return ans


class Scraggle(Board):

    def __init__(self, wordList):
        # initialize the board to the prefilled state (with vowels)

        super().__init__()

        self.board = Board.initBoard()
        self.wordList = wordList               # the raw list of legal scraggle-words
        self.T = self.generateTrie()           # the Trie, build on @wordList

        self.words = []                        # given the board, filtered list of words from wordList, which can be obtained
                                               #    by walking the board using chess king moves (the same letter can be reused)
        self.words_coord = []                  # coordinates (list of tuples) of the words in @words
        self.seen = set()                      # used to track the already generated words from @wordList



    def __str__(self):
        s = ""
        for i in range(6):
            s += " ".join( self.board[i] )
            if i < 5: s += "\n"
        return s


    def createRandomBoard(self):
        self.board = self.generateRandomBoard()


    def generateTrie(self):
        T = Trie(Node('/' ))
        for i, word in enumerate(self.wordList):
            T.insert(word, i )

        print('Trie is ready')
        return T



    def findValidWords(self):
        """
           given the board, and the already build Trie, find all words which can be obtained by king moves
           on the self.board
        """

        self.words = []
        self.words_coord = []
        self.seen = set()

        for i in range(6):
            for j in range(6):
                self.lookup(self.T.root, i,j, '', [])


    def lookup(self, current_Node, a, b, wordPath, coordPath ):
        # look for possible moves from (a,b) on board, and in the children nodes of the current node
        # the wordPath represents the char-path covered so far

        if current_Node.value != None:
            if not wordPath in self.seen:
                self.words.append(  wordPath )
                self.words_coord.append( coordPath )
                self.seen.add( wordPath )
            return

        if (a >= 6 or a < 0 )or( b >= 6 or b < 0)or( len(wordPath) > 15 ):
            # we're out of the board region
            return

        char = self.board[a][b]
        if not char in current_Node.children:
            return


        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx != 0 or dy != 0:
                    self.lookup( current_Node.children[char], a + dx, b + dy, wordPath + char, coordPath + [ (a,b) ] )



    @staticmethod
    def words_start_end_restricted( coords, start_a, start_b, start_c, start_d, end_a, end_b, end_c, end_d ):
        # from a list of coordinates, return those which start and end at the given boxes
        # START box: [ start_a, start_b ]x[start_c, start_d], the same for END box

        ans = []
        for i, coord in enumerate(coords):
            a, b = coord[0] # start coordinate of the word

            if not ( start_a <= a <= start_b and start_c <= b <= start_d ):
                continue

            a, b = coord[-1]
            if not ( end_a <= a <= end_b and end_c <= b <= end_d ):
                continue

            ans.append((i, coord) )

        return ans


    def bestChainOntheBoard(self, output = False):
        # given the board, we find the best (in terms of scraggle score) chain on the board

        candidates = self.findChains()
        if not candidates:
            if output:  print('no chains were found for this board')
            return None, None

        if output:
            print('there are {} chains in total'.format(len(candidates)))

        best_index, best_score = -1, 0

        for i, sampleChain in enumerate( candidates ):
            score = self.computeChainScore( sampleChain )
            if score > best_score:
                best_score = score
                best_index = i

        if output:
            print('best score = ', best_score)
            print('best chain is\n', self.chain_to_str( candidates[best_index] )  )

        return best_score, candidates[ best_index ]



    def findChains(self):
        # find a chain of words in red, blue, green, violet;

        ans = []

        red_blue = Scraggle.words_start_end_restricted(self.words_coord, 0,2,0,2,  1,2,1,2 )
        # list of tuples (index, coord)

        for (i, coord) in red_blue:
            blue_start = coord[-1]

            # find all words which start at the blue_start and end in the green region
            blue_green = Scraggle.words_start_end_restricted(self.words_coord, blue_start[0], blue_start[0], blue_start[1], blue_start[1],
                                               2,3,2,3)

            for (j, coord_1) in blue_green:
                green_start = coord_1[-1]

                # find all words which start at green_start and end in the violet region
                green_violet = Scraggle.words_start_end_restricted(self.words_coord, green_start[0], green_start[0], green_start[1], green_start[1],
                                                     3,4,3,4)

                for (k, coord_2) in green_violet:
                    violet_start = coord_2[-1]

                    # find all words which start at violet_start and end in violet box
                    violet_end = Scraggle.words_start_end_restricted(self.words_coord, violet_start[0],violet_start[0],violet_start[1],violet_start[1],
                                                       3,5,3,5)

                    for (p, coord_3) in violet_end:
                        ans.append( [ (i, coord), (j, coord_1), (k, coord_2), (p, coord_3) ] )


        return ans


    def chain_to_str(self, sampleChain):
        # print the chain, a list of 4 lists of [word_index in words, [coordinates in board]]

        ans = ""
        for (w_ind, word_path) in sampleChain:
            s = Scraggle.getBoardPath(word_path)
            ans += self.words[w_ind]  + ' ' + s + '\n'
        return ans

    @staticmethod
    def scraggleScore(word):
        # compute the scraggle score of the given word

        ans = 0

        # the official scraggle scores for each character
        scores = {'e':1, 'a':1, 'i':1, 'o':1, 'n':1, 'r':1, 't':1, 'l':1, 's':1, 'u':1,
              'd':2, 'g':2,
              'b':3, 'c':3,'m':3,'p':3,
              'f':4, 'h':4, 'v':4, 'w':4, 'y':4,
              'k':5,
              'j':8,  'x':8,
              'q':10, 'z':10 }

        for c in word:
            ans += scores[c]

        return ans

    @staticmethod
    def getDirection(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if (dx, dy) == (0, 1):
            return '\u2192'
        elif (dx, dy) == (0, -1):
            return '\u2190'
        elif (dx, dy) == (1, 0):
            return '\u2193'
        elif (dx, dy) == (-1, 0):
            return '\u2191'
        elif (dx, dy) == (-1, -1):
            return '\u2196'
        elif (dx, dy) == (-1, 1):
            return '\u2197'
        elif (dx, dy) == (1, 1):
            return '\u2198'
        elif (dx, dy) == (1, -1):
            return '\u2199'
        else:
            return ''

    @staticmethod
    def getBoardPath(coords):
        # @coords is a list of tuples
        s = str( coords[0] )
        p1 = coords[0]
        for i in range(1, len(coords)):
            p2 = coords[i]
            s += " " + Scraggle.getDirection(p1, p2)
            p1 = coords[i]

        return s


    def computeChainScore(self, wordChain):
        if not wordChain:
            return 0

        ans = 1
        for (w_ind, _) in wordChain:
            word = self.words[w_ind]
            ans = ans*Scraggle.scraggleScore(word)

        return ans




if __name__ == "__main__":
    chars_consonants = {'y', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z'}
    print('there are {} consonants'.format(len(chars_consonants)))

    raw_list = readDictionary_FromText( )
    print('number of words available =', len(raw_list))

    dd_pruned = removeNonDistinct_Consonants(raw_list, chars_consonants)
    print('there are {} words, which do not have repeating consonants'.format(len(dd_pruned)))

    game = Scraggle(dd_pruned)

    best_score, best_candidate, best_board = 0, None, None

    i = 0

    while True:
        i += 1

        game.createRandomBoard()
        game.findValidWords()

        score, candidate = game.bestChainOntheBoard()

        if score != None and score >= best_score:
            best_score = score
            best_candidate = game.chain_to_str(candidate)
            best_board = Board.board_to_str(game.board)

        if i%1000 == 0:
            print('iteration N ', i )

            if best_score == 0:
                print('\n\nfailed to find a chain', flush = True)
            else:
                print('==============================================')
                print('\n\nbest score = ', best_score,'\n')
                print('best board:')
                print(best_board, '\n' )
                print('best chain:\n')
                print(best_candidate)
                print('==============================================')
                print('\n\n', flush = True)
