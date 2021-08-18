from BaseAI import BaseAI
import time

class PlayerAI(BaseAI):

    def getMove(self, grid):
        grid.depth = 0
        self.max_depth = 4
        self.start_time = time.clock()
        self.time_limit = 0.18
        move = self.decision(grid)
        return move

    def isTerminal(self, grid):
        return not grid.canMove()

    def timeExceeded(self):
        if self.time_limit<time.clock()-self.start_time:
            return True
        return False

    def depthExceeded(self, grid):
        if self.max_depth<grid.depth:
            return True
        return False
    
    def eval(self, grid):
        r = 2
        board1 = [[r**16, r**15, r**14, r**13],
                 [r**9, r**10, r**11, r**12],
                 [r**8, r**7, r**6, r**5],
                 [r**1, r**2, r**3, r**4]
                ]
        result1 = 0
        for i in range(4):
            for j in range(4):
                result1 = result1 + grid.map[i][j]*board1[i][j]
        board2 = [[r**13, r**14, r**15, r**16],
                 [r**12, r**11, r**10, r**9],
                 [r**5, r**6, r**7, r**8],
                 [r**4, r**3, r**2, r**1]
                ]
        result2 = 0
        for i in range(4):
            for j in range(4):
                result2 = result2 + grid.map[i][j]*board2[i][j]   
        board3 = [[r**4, r**5, r**12, r**13],
                 [r**3, r**6, r**11, r**14],
                 [r**2, r**7, r**10, r**15],
                 [r**1, r**8, r**9, r**16]
                ]
        result3 = 0
        for i in range(4):
            for j in range(4):
                result3 = result3 + grid.map[i][j]*board3[i][j]
        board4 = [[r**13, r**12, r**5, r**4],
                 [r**14, r**11, r**6, r**3],
                 [r**15, r**10, r**7, r**2],
                 [r**16, r**9, r**8, r**1]
                ]
        result4 = 0
        for i in range(4):
            for j in range(4):
                result4 = result4 + grid.map[i][j]*board4[i][j]
        board5 = [[r**1, r**2, r**3, r**4],
                 [r**8, r**7, r**6, r**5],
                 [r**9, r**10, r**11, r**12],
                 [r**16, r**15, r**14, r**13]
                ]
        result5 = 0
        for i in range(4):
            for j in range(4):
                result5 = result5 + grid.map[i][j]*board5[i][j]
        board6 = [[r**4, r**3, r**2, r**1],
                 [r**5, r**6, r**7, r**8],
                 [r**12, r**11, r**10, r**9],
                 [r**13, r**14, r**15, r**16]
                ]
        result6 = 0
        for i in range(4):
            for j in range(4):
                result6 = result6 + grid.map[i][j]*board6[i][j] 
        board7 = [[r**1, r**8, r**9, r**16],
                 [r**2, r**7, r**10, r**15],
                 [r**3, r**6, r**11, r**14],
                 [r**4, r**5, r**12, r**13]
                ]
        result7 = 0
        for i in range(4):
            for j in range(4):
                result7 = result7 + grid.map[i][j]*board7[i][j]
        board8 = [[r**16, r**9, r**8, r**1],
                 [r**15, r**10, r**7, r**2],
                 [r**14, r**11, r**6, r**3],
                 [r**13, r**12, r**5, r**4]
                ]
        result8 = 0
        for i in range(4):
            for j in range(4):
                result8 = result8 + grid.map[i][j]*board8[i][j]
        return max(result1, result2, result3, result4, result5, result6, result7, result8)
        
    def maximize(self, grid, alpha, beta):
        if self.isTerminal(grid) or self.timeExceeded() or self.depthExceeded(grid):
            return None, self.eval(grid) 
        maxChild, maxUtility = None, -float('Inf')
        for x in grid.getAvailableMoves():
            child = grid.clone()
            child.depth = grid.depth + 1
            child.move(x)
            _, utility = self.minimize(child, alpha, beta)
            if utility>maxUtility:
                maxChild, maxUtility = child, utility
            if maxUtility>=beta:
                break
            if maxUtility>alpha:
                alpha = maxUtility
            if self.timeExceeded():
                break
        return maxChild, maxUtility

    def minimize(self, grid, alpha, beta):
        if self.isTerminal(grid) or self.timeExceeded() or self.depthExceeded(grid):
            return None, self.eval(grid)
        minChild, minUtility = None, float('Inf')
        for value in [2,4]:
            for p in grid.getAvailableCells():
                child = grid.clone()
                child.depth = grid.depth + 1
                child.insertTile(p,value)
                _, utility = self.maximize(child, alpha, beta)
                if utility<minUtility:
                    minChild, minUtility = child, utility
                if minUtility<=alpha:
                    break
                if minUtility<beta:
                    beta = minUtility
                if self.timeExceeded():
                    break
            if self.timeExceeded():
                break
        return minChild, minUtility

    def decision(self, grid):
        child, _ = self.maximize(grid, -float('Inf'), float('Inf'))
        for x in grid.getAvailableMoves():
            gridCopy = grid.clone()
            gridCopy.move(x)
            if gridCopy.map == child.map:
                return x
        return None