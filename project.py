"""
Author : Mukesh T P
"""


class Graph:
    """
    Graph class for initializing and managing a graph.
    
    Attributes:
        graph: Dictionary where keys represent nodes, and values are lists of nodes connected to the key node.
        weight: Dictionary where keys represent nodes, and values are lists of weights corresponding to edges connected to the key node.
        heuristic: Dictionary where keys represent nodes, and values are heuristic values from the source to the goal.
    """

    def __init__(self):
        """
        Initializes the graph, weight, and heuristic dictionaries.
        """
        self.graph = {}
        self.weight = {}
        self.heuristic = {}

    def addEdge(self, o, d, w = 1):
        """
        Adds an edge between two points in the graph.

        Parameters:
            o: Origin/start/current node.
            d: Destination node.
            w: Weight of the edge (default = 1).
        """
        if o not in self.graph:
            self.graph[o] = []
            self.weight[o] = []
            self.heuristic[o] = 100
        if d not in self.graph:
            self.graph[d] = []
            self.weight[d] = []
            self.heuristic[d] = 100
        self.graph[o].append(d)
        self.weight[o].append(w)
        combined = sorted(zip(self.graph[o], self.weight[o]), key=lambda x: x[0])
        self.graph[o], self.weight[o] = map(list, zip(*combined))
        self.graph[d].append(o)
        self.weight[d].append(w)
        combined = sorted(zip(self.graph[d], self.weight[d]), key=lambda x: x[0])
        self.graph[d], self.weight[d] = map(list, zip(*combined))

    def addHeuristics(self, o, h):
        """
        Adds heuristic value to the point mentioned.

        Parameters:
            o: Origin/start/current node.
            h: Heuristic value (default value = 100).
        """
        self.heuristic[o] = h
    """
    def __str__(self):
    
        #Prints the graph, weight and hueristic
        
        return f"{self.graph}\n{self.weight}\n{self.heuristic}"
    """

class Algorithm:

    """
    This class contains searching techniques that can be used on a Graph.
    Parameters:
        g : graph
        o : origin
        d : destination
        w : weight (default value = 1)
        h : heuristics (default value = 100)
    """

    def DFS(self, g, o, d):
        """
        This implements Depth First Search on a given graph.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        path = []
        visited = set()
        stack = [o]
        while stack:
            node = stack.pop()
            if node == d:
                return path + [node]
            if node not in visited:
                path.append(node)
                visited.add(node)
                for neighbor in sorted(g.graph[node], reverse=True):
                    if neighbor not in visited:
                        stack.append(neighbor)
        return path

    def BFS(self, g, o, d):
        """
        This implements Breadth First Search on a given graph.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        path = []
        visited = set()
        stack = [o]
        while stack:
            node = stack.pop(0)
            if node == d:
                return path + [node]
            if node not in visited:
                path.append(node)
                visited.add(node)
                for neighbor in g.graph[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
        return None

    def BMS(self, g, o, d):
        """
        This implements British Museum Search on a given graph.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        paths = []
        stack = [(o, [o])]
        while stack:
            node, path = stack.pop()
            if node == d:
                paths.append(path)
            else:
                for neighbor in g.graph[node]:
                    if neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))
        return paths

    # def HC(self, g, o, d): This one isn't correct
    #     path = []
    #     visited = set()
    #     stack = [o]
    #     while stack:
    #         node = stack.pop()
    #         if node == d:
    #             return path + [node]
    #         if node not in visited:
    #             path.append(node)
    #             visited.add(node)
    #             for neighbor in sorted(g.graph[node], key=lambda x: g.heuristic[x]):
    #                 if neighbor not in visited:
    #                     stack.append(neighbor)
    #     return None
    
    def HC(self, g, o, d):
        """
        This implements Hill Climbing on a given graph.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        path = []
        visited = set()
        node = o
        while node != d:
            path.append(node)
            visited.add(node)
            neighbors = g.graph[node]
            neighbor_heuristics = [g.heuristic[neighbor] for neighbor in neighbors]
            best_neighbor = neighbors[neighbor_heuristics.index(min(neighbor_heuristics))]
            if best_neighbor in visited:
                return path
            node = best_neighbor
        path.append(d)
        return path
    
    def BS(self, g, o, d, bw=1):
        """
        This implements Beam Search on a given graph.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
            bw : determines the beam width required (default value = 1)
        """
        beam = [(g.heuristic[o], (o, [o]))]
        while beam:
            beam.sort(key=lambda x: x[0])
            best_paths = beam[:bw]
            beam = []
            for misc, (node, path) in best_paths:
                if node == d:
                    return path
                for neighbor in g.graph[node]:
                    if neighbor not in path:
                        heuristic_score = g.heuristic[neighbor]
                        new_path = path + [neighbor]
                        beam.append((heuristic_score, (neighbor, new_path)))
        return None

    def Oracle(self, g, o, d):
        """
        Oracle search performing an exhaustive search to find all possible paths.
        Returns a list of tuples, each containing a path and its cost.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        all_paths = []
        stack = [(o, [], 0)]  # (node, path, cost)
        while stack:
            current, path, cost = stack.pop()
            if current == d:
                all_paths.append((path + [current], cost))
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        stack.append((neighbor, path + [current], cost + weight))
        return all_paths

    def BB(self, g, o, d):
        """
        Branch and Bound algorithm to find the optimal path.
        Returns the optimal path and its cost.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        best_path = None
        best_cost = float('inf')  # Initialize with positive infinity

        # Priority queue implemented as a list of tuples (cost, node, path)
        priority_queue = [(0, o, [])]

        while priority_queue:
            # Find the path with the lowest cost in the priority queue
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] < priority_queue[min_index][0]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)
            if current == d:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        # Add the neighbor to the priority queue with updated cost
                        priority_queue.append((cost + weight, neighbor, path + [current]))
                

        return best_path, best_cost

    def EL(self, g, o, d):
        """
        Branch and Bound algorithm with an extended list.
        Returns the optimal path and its cost.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        best_path = None
        best_cost = float('inf')  # Initialize with positive infinity

        # Priority queue implemented as a list of tuples (cost, node, path)
        priority_queue = [(0, o, [])]

        # Extended list to keep track of visited nodes
        extended_list = {node: False for node in g.graph}

        while priority_queue:
            # Find the path with the lowest cost in the priority queue
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] < priority_queue[min_index][0]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)
            print(cost, current, path)
            # Mark the current node as visited
            extended_list[current] = True
            print
            if current == d:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if not extended_list[neighbor]:
                        # Add the neighbor to the priority queue with updated cost
                        priority_queue.append((cost + weight, neighbor, path + [current]))

        return best_path, best_cost
    
    def EH(self, g, o, d):
        """
        Branch and Bound algorithm with estimated heuristics.
        Returns the optimal path and its cost.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        best_path = None
        best_cost = float('inf')  # Initialize with positive infinity

        # Priority queue implemented as a list of tuples (cost, node, path)
        priority_queue = [(0, o, [])]

        while priority_queue:
            # Find the path with the lowest cost in the priority queue
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] + g.heuristic[priority_queue[i][1]] < priority_queue[min_index][0] + g.heuristic[priority_queue[min_index][1]]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)

            if current == d:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if neighbor not in path:
                        # Add the neighbor to the priority queue with updated cost
                        priority_queue.append((cost + weight, neighbor, path + [current]))

        return best_path, best_cost
    
    def Astar(self, g, o, d):
        """
        Branch and Bound algorithm with extended list and estimated heuristics.
        Returns the optimal path and its cost.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        best_path = None
        best_cost = float('inf')  # Initialize with positive infinity

        # Priority queue implemented as a list of tuples (cost, node, path)
        priority_queue = [(0, o, [])]

        # Extended list to keep track of visited nodes
        extended_list = {node: False for node in g.graph}

        while priority_queue:
            # Find the path with the lowest cost in the priority queue
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] + g.heuristic[priority_queue[i][1]] < priority_queue[min_index][0] + g.heuristic[priority_queue[min_index][1]]:
                    min_index = i
            cost, current, path = priority_queue.pop(min_index)
        
            # Use the extended list to track visited nodes for the current path
            visited = set(path)

            # Mark the current node as visited in the global set
            extended_list[current] = True

            if current == d:
                if cost < best_cost:
                    best_path = path + [current]
                    best_cost = cost
            else:
                for neighbor, weight in zip(g.graph[current], g.weight[current]):
                    if not extended_list[neighbor] and neighbor not in visited:
                        # Add the neighbor to the priority queue with updated cost
                        priority_queue.append((cost + weight, neighbor, path + [current]))

        return best_path, best_cost
    
    def BestFirstSearch(self, g, o, d):
        """
        Best-First Search algorithm.
        Returns the optimal path.
        Parameters:
            g : is the object of class Graph
            o : origin/start/current node
            d : destination node
        """
        best_path = None

        # Priority queue implemented as a list of tuples (heuristic, node, path)
        priority_queue = [(g.heuristic[o], o, [])]

        while priority_queue:
            # Find the path with the lowest heuristic value in the priority queue
            min_index = 0
            for i in range(1, len(priority_queue)):
                if priority_queue[i][0] < priority_queue[min_index][0]:
                    min_index = i
            heuristic, current, path = priority_queue.pop(min_index)

            if current == d:
                # Destination reached, update best_path
                best_path = path + [current]
            else:
                for neighbor in g.graph[current]:
                    if neighbor not in path:
                        # Add the neighbor to the priority queue with updated heuristic
                        priority_queue.append((g.heuristic[neighbor], neighbor, path + [current]))

        return best_path


    

choice = input("Click Enter to continue with default values, else enter 1")
g = Graph()

if choice == '':
    g.addEdge('S','A',3)
    g.addEdge('S','B',5)
    g.addEdge('A','B',4)
    g.addEdge('A','D',3)
    g.addEdge('D','G',5)
    g.addEdge('B','C',4)
    g.addEdge('C','E',6)
    g.addHeuristics('S',10)
    g.addHeuristics('A',7)
    g.addHeuristics('B',6)
    g.addHeuristics('C',7)
    g.addHeuristics('D',5)
    g.addHeuristics('E',4)
    g.addHeuristics('G',0)
else:
    pass


#print(g)
#print(Algorithm().HC(g, 'S', 'G'))

# results = Algorithm().Oracle(g, 'S', 'G')
# for path, cost in results:
#     print(f"Path: {path}, Cost: {cost}")

# best_path, best_cost = Algorithm().BestFirstSearch(g, 'S', 'G')
# print(f"Optimal Path: {best_path}, Optimal Cost: {best_cost}")

best_path = Algorithm().Oracle(g, 'S', 'G')
print(f"Optimal Path: {best_path}")

"""
Author : Mukesh T P
"""
