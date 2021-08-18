import queue as Q
import time
import resource
import sys
import math

class PuzzleState(object):

    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        if n*n != len(config) or n < 2:
            raise Exception("the length of config is not correct!")
        self.n = n
        self.cost = cost
        self.parent = parent
        self.action = action
        self.dimension = n
        self.config = config
        self.children = []
        for i, item in enumerate(self.config):
            if item == 0:
                self.blank_row = i // self.n
                self.blank_col = i % self.n
                break

    def __lt__(self, other):
        return self.cost < other.cost

    def display(self):
        for i in range(self.n):
            line = []
            offset = i * self.n
            for j in range(self.n):
                line.append(self.config[offset + j])
            print(line)

    def move_left(self):
        if self.blank_col == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):
        if self.blank_col == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):
        if self.blank_row == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):
        if self.blank_row == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):
        if len(self.children) == 0:
            up_child = self.move_up()
            if up_child is not None:
                self.children.append(up_child)
            down_child = self.move_down()
            if down_child is not None:
                self.children.append(down_child)
            left_child = self.move_left()
            if left_child is not None:
                self.children.append(left_child)
            right_child = self.move_right()
            if right_child is not None:
                self.children.append(right_child)
        return self.children


class PuzzleSolver(object):
    
    def __init__(self, initial_state, goal_state_config):
        self.initial_state = initial_state
        self.goal_state_config = goal_state_config
        self.path_to_goal = []
        self.search_depth = 0
        self.max_search_depth = 0
    
    def writeOutput(self):
        with open('output.txt','w') as f:
            f.write('path_to_goal: ')
            f.write(str(self.path_to_goal))
            f.write('\n')
            f.write('cost_of_path: ')
            f.write(str(self.cost_of_path))
            f.write('\n')
            f.write('nodes_expanded: ')
            f.write(str(self.nodes_expanded))
            f.write('\n')
            f.write('search_depth: ')
            f.write(str(self.search_depth))
            f.write('\n')
            f.write('max_search_depth: ')
            f.write(str(self.max_search_depth))
            f.write('\n')
            f.write('running_time: ')
            f.write(str(round(self.running_time,8)))
            f.write('\n')
            f.write('max_ram_usage: ')
            f.write(str(round(self.max_ram_usage,8)))  

    def bfs_search(self):
        start_time = time.time()
        frontier = [Q.Queue(),{}]
        frontier[0].put(self.initial_state)
        frontier[1][self.initial_state.config] = self.initial_state
        explored = [[],{}]
        while not frontier[0].empty():
            state = frontier[0].get()
            del frontier[1][state.config]
            explored[0].append(state)
            explored[1][state.config] = state
            self.max_search_depth = max(state.cost, self.max_search_depth)
            if self.test_goal(state):
                while state.parent!=None:
                    self.path_to_goal.append(state.action)
                    state = state.parent
                    self.search_depth = self.search_depth+1
                self.nodes_expanded = len(explored[0])-1
                self.path_to_goal = self.path_to_goal[::-1]
                self.cost_of_path = len(self.path_to_goal)
                self.max_search_depth = self.max_search_depth+1
                self.running_time = time.time() - start_time
                self.max_ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1e-6
                return True
            for child in state.expand():
                if (child.config not in frontier[1]) and (child.config not in explored[1]):
                    frontier[0].put(child)
                    frontier[1][child.config] = child
        return False

    def dfs_search(self):
        start_time = time.time()
        frontier = [Q.LifoQueue(),{}]
        frontier[0].put(self.initial_state)
        frontier[1][self.initial_state.config] = self.initial_state
        explored = [[],{}]
        while not frontier[0].empty():
            state = frontier[0].get()
            del frontier[1][state.config]
            explored[0].append(state)
            explored[1][state.config] = state
            self.max_search_depth = max(state.cost, self.max_search_depth)
            if self.test_goal(state):
                while state.parent!=None:
                    self.path_to_goal.append(state.action)
                    state = state.parent
                    self.search_depth = self.search_depth+1
                self.nodes_expanded = len(explored[0])-1
                self.path_to_goal = self.path_to_goal[::-1]
                self.cost_of_path = len(self.path_to_goal)
                self.running_time = time.time() - start_time
                self.max_ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1e-6
                return True
            for child in state.expand()[::-1]:
                if (child.config not in frontier[1]) and (child.config not in explored[1]):
                    frontier[0].put(child)
                    frontier[1][child.config] = child
        return False

    def a_star_search(self):
        start_time = time.time()
        frontier = [Q.PriorityQueue(),{},{}]
        frontier[0].put((self.calculate_total_cost(self.initial_state), self.initial_state))
        frontier[1][self.initial_state.config] = self.initial_state
        frontier[2][self.initial_state.config] = self.calculate_total_cost(self.initial_state)
        explored = [[],{}]
        while not frontier[0].empty():
            cost, state = frontier[0].get()
            if state.config in frontier[2]:
                if frontier[2][state.config]==cost:
                    del frontier[1][state.config]
                    del frontier[2][state.config]
                    explored[0].append(state)
                    explored[1][state.config] = state
                    self.max_search_depth = max(state.cost, self.max_search_depth)
                    if self.test_goal(state):
                        while state.parent!=None:
                            self.path_to_goal.append(state.action)
                            state = state.parent
                            self.search_depth = self.search_depth+1
                        self.nodes_expanded = len(explored[0])-1
                        self.path_to_goal = self.path_to_goal[::-1]
                        self.cost_of_path = len(self.path_to_goal)
                        self.running_time = time.time() - start_time
                        self.max_ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1e-6
                        return True
                    for child in state.expand()[::-1]:
                        if (child.config not in frontier[1]) and (child.config not in explored[1]):
                            cost = self.calculate_total_cost(child)
                            frontier[0].put((cost, child))
                            frontier[1][child.config] = child
                            frontier[2][child.config] = cost
                        else:
                            if child.config in frontier[1]:
                                if self.calculate_total_cost(child)<frontier[2][child.config]:
                                    new_cost = self.calculate_total_cost(child)
                                    frontier[0].put((new_cost, child))
                                    frontier[2][child.config] = new_cost
        return False

    def calculate_total_cost(self, puzzle_state):
        g = puzzle_state.cost
        h = puzzle_state.blank_row+puzzle_state.blank_col
        return g+h
            
    def test_goal(self, puzzle_state):
        if puzzle_state.config==self.goal_state_config:
            return True
        return False

def main():
    s_a = sys.argv[1].lower()
    i_s = sys.argv[2].split(",")
    i_s = tuple(map(int, i_s))
    size = int(math.sqrt(len(i_s)))
    initial_state = PuzzleState(i_s, size)
    goal_state_config = (0,1,2,3,4,5,6,7,8)
    if s_a == "bfs":
        puzzle_solver = PuzzleSolver(initial_state, goal_state_config)
        if puzzle_solver.bfs_search():
            puzzle_solver.writeOutput()
    elif s_a == "dfs":
        puzzle_solver = PuzzleSolver(initial_state, goal_state_config)
        if puzzle_solver.dfs_search():
            puzzle_solver.writeOutput()
    elif s_a == "ast":
        puzzle_solver = PuzzleSolver(initial_state, goal_state_config)
        if puzzle_solver.a_star_search():
            puzzle_solver.writeOutput()
    else:
        print("Enter valid command arguments!")

if __name__ == '__main__':
    main()