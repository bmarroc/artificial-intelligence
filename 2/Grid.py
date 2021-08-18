from copy import deepcopy

directionVectors = (UP_VEC, DOWN_VEC, LEFT_VEC, RIGHT_VEC) = ((-1, 0), (1, 0), (0, -1), (0, 1))
vecIndex = [UP, DOWN, LEFT, RIGHT] = range(4)

class Grid:

    def __init__(self, size = 4):
        self.size = size
        self.map = [[0] * self.size for i in range(self.size)]

    def clone(self):
        gridCopy = Grid()
        gridCopy.map = deepcopy(self.map)
        gridCopy.size = self.size
        return gridCopy

    def insertTile(self, pos, value):
        self.setCellValue(pos, value)

    def setCellValue(self, pos, value):
        self.map[pos[0]][pos[1]] = value

    def getAvailableCells(self):
        cells = []
        for x in range(self.size):
            for y in range(self.size):
                if self.map[x][y] == 0:
                    cells.append((x,y))
        return cells

    def getMaxTile(self):
        maxTile = 0
        for x in range(self.size):
            for y in range(self.size):
                maxTile = max(maxTile, self.map[x][y])
        return maxTile

    def canInsert(self, pos):
        return self.getCellValue(pos) == 0

    def move(self, dir):
        dir = int(dir)
        if dir == UP:
            return self.moveUD(False)
        if dir == DOWN:
            return self.moveUD(True)
        if dir == LEFT:
            return self.moveLR(False)
        if dir == RIGHT:
            return self.moveLR(True)

    def moveUD(self, down):
        r = range(self.size -1, -1, -1) if down else range(self.size)
        moved = False
        for j in range(self.size):
            cells = []
            for i in r:
                cell = self.map[i][j]
                if cell != 0:
                    cells.append(cell)
            self.merge(cells)
            for i in r:
                value = cells.pop(0) if cells else 0
                if self.map[i][j] != value:
                    moved = True
                self.map[i][j] = value
        return moved

    def moveLR(self, right):
        r = range(self.size - 1, -1, -1) if right else range(self.size)
        moved = False
        for i in range(self.size):
            cells = []
            for j in r:
                cell = self.map[i][j]
                if cell != 0:
                    cells.append(cell)
            self.merge(cells)
            for j in r:
                value = cells.pop(0) if cells else 0
                if self.map[i][j] != value:
                    moved = True
                self.map[i][j] = value
        return moved

    def merge(self, cells):
        if len(cells) <= 1:
            return cells
        i = 0
        while i < len(cells) - 1:
            if cells[i] == cells[i+1]:
                cells[i] *= 2
                del cells[i+1]
            i += 1

    def canMove(self, dirs = vecIndex):
        checkingMoves = set(dirs)
        for x in range(self.size):
            for y in range(self.size):
                if self.map[x][y]:
                    for i in checkingMoves:
                        move = directionVectors[i]
                        adjCellValue = self.getCellValue((x + move[0], y + move[1]))
                        if adjCellValue == self.map[x][y] or adjCellValue == 0:
                            return True
                elif self.map[x][y] == 0:
                    return True
        return False

    def getAvailableMoves(self, dirs = vecIndex):
        availableMoves = []
        for x in dirs:
            gridCopy = self.clone()
            if gridCopy.move(x):
                availableMoves.append(x)
        return availableMoves

    def crossBound(self, pos):
        return pos[0] < 0 or pos[0] >= self.size or pos[1] < 0 or pos[1] >= self.size

    def getCellValue(self, pos):
        if not self.crossBound(pos):
            return self.map[pos[0]][pos[1]]
        else:
            return None

if __name__ == '__main__':
    g = Grid()
    g.map[0][0] = 2
    g.map[1][0] = 2
    g.map[3][0] = 4
    while True:
        for i in g.map:
            print(i)
        print(g.getAvailableMoves())
        v = input()
        g.move(v)
