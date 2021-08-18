import sys
import copy

class Sudoku(object):
            
    def __init__(self):
        self.csp_1 = [['A1','A2','A3','A4','A5','A6','A7','A8','A9'], 
                      ['B1','B2','B3','B4','B5','B6','B7','B8','B9'],
                      ['C1','C2','C3','C4','C5','C6','C7','C8','C9'],
                      ['D1','D2','D3','D4','D5','D6','D7','D8','D9'],
                      ['E1','E2','E3','E4','E5','E6','E7','E8','E9'],
                      ['F1','F2','F3','F4','F5','F6','F7','F8','F9'],
                      ['G1','G2','G3','G4','G5','G6','G7','G8','G9'],
                      ['H1','H2','H3','H4','H5','H6','H7','H8','H9'],
                      ['I1','I2','I3','I4','I5','I6','I7','I8','I9']]
        self.csp_2 = [['A1','B1','C1','D1','E1','F1','G1','H1','I1'], 
                      ['A2','B2','C2','D2','E2','F2','G2','H2','I2'],
                      ['A3','B3','C3','D3','E3','F3','G3','H3','I3'],
                      ['A4','B4','C4','D4','E4','F4','G4','H4','I4'],
                      ['A5','B5','C5','D5','E5','F5','G5','H5','I5'],
                      ['A6','B6','C6','D6','E6','F6','G6','H6','I6'],
                      ['A7','B7','C7','D7','E7','F7','G7','H7','I7'],
                      ['A8','B8','C8','D8','E8','F8','G8','H8','I8'],
                      ['A9','B9','C9','D9','E9','F9','G9','H9','I9']]
        self.csp_3 = [['A1','B1','C1','A2','B2','C2','A3','B3','C3'], 
                      ['A4','B4','C4','A5','B5','C5','A6','B6','C6'],
                      ['A7','B7','C7','A8','B8','C8','A9','B9','C9'],
                      ['D1','E1','F1','D2','E2','F2','D3','E3','F3'],
                      ['D4','E4','F4','D5','E5','F5','D6','E6','F6'],
                      ['D7','E7','F7','D8','E8','F8','D9','E9','F9'],
                      ['G1','H1','I1','G2','H2','I2','G3','H3','I3'],
                      ['G4','H4','I4','G5','H5','I5','G6','H6','I6'],
                      ['G7','H7','I7','G8','H8','I8','G9','H9','I9']]
        self.arcs = []
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    if j!=k:
                        self.arcs.append((self.csp_1[i][j],self.csp_1[i][k]))
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    if j!=k:
                        self.arcs.append((self.csp_2[i][j],self.csp_2[i][k]))
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    if j!=k:
                        self.arcs.append((self.csp_3[i][j],self.csp_3[i][k]))
        self.arcs = list(dict.fromkeys(self.arcs))
        self.variables = ['A1','A2','A3','A4','A5','A6','A7','A8','A9',
                          'B1','B2','B3','B4','B5','B6','B7','B8','B9',
                          'C1','C2','C3','C4','C5','C6','C7','C8','C9',
                          'D1','D2','D3','D4','D5','D6','D7','D8','D9',
                          'E1','E2','E3','E4','E5','E6','E7','E8','E9',
                          'F1','F2','F3','F4','F5','F6','F7','F8','F9',
                          'G1','G2','G3','G4','G5','G6','G7','G8','G9',
                          'H1','H2','H3','H4','H5','H6','H7','H8','H9',
                          'I1','I2','I3','I4','I5','I6','I7','I8','I9']
        self.neighbors = {}
        for x in self.variables:
            self.neighbors[x] = []
        for (key,value) in self.arcs:
            self.neighbors[key].append(value)
    
    def set_board(self, board):
        self.board = board
        
    def get_board(self):
        return self.board
            
    def revise(self, xi, xj):
        revised = False
        for z in self.board[xi]:
            delete = True
            for t in self.board[xj]:
                if z!=t:
                    delete = False
                    break
            if delete:
                self.board[xi].remove(z)
                revised = True
        return revised
            
    def AC3(self):
        queue = self.arcs.copy()
        while len(queue)!=0:
            xi, xj = queue.pop(0)
            if self.revise(xi, xj):
                if len(self.board[xi])==0:
                    return False
                for xk in self.neighbors[xi]:
                    if xk!=xj:
                        queue.append((xk,xi))
        return True

    def select_unassigned_variable(self, assignment):
        fewest_legal_values = ('', 10)
        for (variable,domain) in assignment.items():
            if 1<len(domain) and len(domain)<fewest_legal_values[1]:
                fewest_legal_values = (variable, len(domain))
        return fewest_legal_values

    def order_domain_values(self, assignment, x):
        least_constraining_values = []
        for value in assignment[x]:
            fewest_legal_values = ('', 10)
            for neighbor in self.neighbors[x]:
                if value in assignment[neighbor] and len(assignment[neighbor])-1<fewest_legal_values[1]:
                    fewest_legal_values = (neighbor, len(assignment[neighbor])-1)
            least_constraining_values.append((value,fewest_legal_values[1]))
        least_constraining_values.sort(key=lambda z:z[1], reverse=True)
        return least_constraining_values

    def is_complete(self, assignment):            
        for domain in assignment.values():
            if len(domain)!=1:
                return False
        return True

    def is_consistent(self, x, value, assignment):            
        for neighbor in self.neighbors[x]:
            if value in assignment[neighbor] and len(assignment[neighbor])-1==0:
                return False
        return True

    def inference(self, x, value, assignment):            
        queue = []
        for neighbor in self.neighbors[x]:
            queue.append((neighbor, value))
        while len(queue)!=0:
            neighbor, value = queue.pop(0)
            if value in assignment[neighbor]:
                assignment[neighbor].remove(value)
                if len(assignment[neighbor])==0:
                    return False , {}
                if len(assignment[neighbor])==1:
                    for neighbor_ in self.neighbors[neighbor]:
                        queue.append((neighbor_, assignment[neighbor][0]))
        return True, assignment

    def backtrack(self, assignment):
        if self.is_complete(assignment):
            return True, assignment       
        x, _ = self.select_unassigned_variable(assignment)
        for (value,_) in self.order_domain_values(assignment, x):
            if self.is_consistent(x, value, assignment):
                assignment1 = copy.deepcopy(assignment)
                assignment1[x] = [value]
                result_inference, assignment2 = self.inference(x, value, assignment1)
                if result_inference:
                    result_backtrack, assignment3 = self.backtrack(assignment2)
                    if result_backtrack:
                        return True, assignment3
            assignment[x].remove(value)
        return False, self.board
    
    def BTS(self):
        _, assignment = self.backtrack(copy.deepcopy(self.board)) 
        self.set_board(assignment)

    def is_solved(self):
        for domain in self.board.values():
            if len(domain)!=1:
                return False
        return True

    def solution(self):
        board = ''
        for (variable,domain) in self.board.items():
                board = board + str(domain[0])
        return board
        
    def print_sudoku(self):
        board = ''
        for (variable,domain) in self.board.items():
            if len(domain)==1:
                board = board + str(domain[0])
            else:
                board = board + '0'
        print("+" + "---+"*9)
        for row in range(9):
            print(("|" + " {}   {}   {} |"*3).format(*[x for x in board[row*9:(row+1)*9]]))
            if row % 3 == 2:
                print("+" + "---+"*9)
            else:
                print("+" + "   +"*9)
                
    def print_sudoku_domains(self):
        print(self.board)
                

def main():
    sudoku = Sudoku()
    board = {}
    initial_board = sys.argv[1]
    for i in range(len(initial_board)):
        if int(initial_board[i])!=0:
            board[sudoku.variables[i]] = [int(initial_board[i])]
        else:
            board[sudoku.variables[i]] = [1,2,3,4,5,6,7,8,9]
    sudoku.set_board(board)
    sudoku.AC3()
    if sudoku.is_solved():
        with open('output.txt','w') as f:
            f.write(sudoku.solution())
            f.write(' ')
            f.write('AC3')    
        return
    sudoku.BTS()
    if sudoku.is_solved():
        with open('output.txt','w') as f:
            f.write(sudoku.solution())
            f.write(' ')
            f.write('BTS')    
        return

if __name__ == '__main__':
    main()