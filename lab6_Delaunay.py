import math
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

class Point:   # we define class point in order to easier store the cartesian coordonates
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

big_circumcenter, big_radius = Point(0,0), 0

class Triangle:  # we use this class to represent the triangles of the triangulation, stroing the 3 vertices,
#additinaly, we have the methods we will use in order to compute if a point is inside the circumcircle

    def __init__(self, p1, p2, p3):
        self.vertices = [p1, p2, p3]

    def midpoint(self, p1, p2):
        return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)

    def perpendicular_bisector(self, p1, p2): # we use this method to compute the perpendicualr bisector of two of the vertices
        mid = self.midpoint(p1, p2)

        if p2.x - p1.x == 0: 
            slope = None
        else:
            slope = (p2.y - p1.y) / (p2.x - p1.x)
        if slope is None:
            perp_slope = 0
        elif slope == 0:  
            perp_slope = None
        else:
            perp_slope = -1 / slope

        #in the end we return the middle point and slope, all that we need to charcterize a perp. bisect.
        return mid, perp_slope
    
    def line_intersection(self, mid1, slope1, mid2, slope2): # we use this funct to find intersection of two points, used to find center of the cirumcircle

        if slope1 is None: 
            x = mid1.x
            y = slope2 * (x - mid2.x) + mid2.y

        elif slope2 is None:  
            x = mid2.x
            y = slope1 * (x - mid1.x) + mid1.y
        elif slope1 == slope2:
            return None
        else:
            x = (slope2 * mid2.x - slope1 * mid1.x + mid1.y - mid2.y) / (slope2 - slope1)
            y = slope1 * (x - mid1.x) + mid1.y

        return Point(x, y)
    
    def compute_circumcenter_and_radius(self):
        A, B, C = self.vertices

        #computing two of the perp. bisectors and the finding their intersectin, i.e. the circumcenter
        #afterwars we compute the distanvce from the circumcenter to a point to determine the radius
        mid1, slope1 = self.perpendicular_bisector(A, B)
        mid2, slope2 = self.perpendicular_bisector(B, C)
        
        circumcenter = self.line_intersection(mid1, slope1, mid2, slope2)
        circumradius = math.sqrt((circumcenter.x - A.x)**2 + (circumcenter.y - A.y)**2)

        #returing the center and the radius, in order to characterize the circumcircle
        return circumcenter, circumradius

    def contains_point_in_circumcircle(self, p):
        circumcenter, circumradius = self.compute_circumcenter_and_radius()

        # we simply check if the point respects the inequality of the circle to find if it's inside it
        distance_squared = (p.x - circumcenter.x)**2 + (p.y - circumcenter.y)**2
        return distance_squared <= circumradius**2
     
    def __repr__(self):
        return f"Triangle({self.vertices[0]}, {self.vertices[1]}, {self.vertices[2]})"

def delaunay_triangulation(points, epsilon=1e-7):
    
    #if two point coincide or are really really close, we will eliminate them, as they only affect our triangulation 
    def is_coincident(p1, p2):
        return abs(p1.x - p2.x) < epsilon and abs(p1.y - p2.y) < epsilon

    unique_points = []
    for p in points:
        if not any(is_coincident(p, existing_point) for existing_point in unique_points):
            unique_points.append(p)

    points = unique_points


    def collinear(p1, p2, p3, epsilon=1e-7):
        return abs((p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)) < epsilon

    points.sort(key=lambda p: p.x)


    min_x = min(p.x for p in points)
    max_x = max(p.x for p in points)
    min_y = min(p.y for p in points)
    max_y = max(p.y for p in points)
    delta_x = max_x - min_x
    delta_y = max_y - min_y
    delta_max = max(delta_x, delta_y)
    big_triangle = Triangle(
        Point(min_x - 7 * delta_max, min_y - delta_max),
        Point(min_x + delta_max, max_y + 7 * delta_max),
        Point(max_x + 7 * delta_max, min_y - delta_max)
    )

    #we first compute a big triangle that contains all the given points, for that we compute the min and max of the x and y coordonates of the points
    min_x = min(p.x for p in points)
    max_x = max(p.x for p in points)
    min_y = min(p.y for p in points)
    max_y = max(p.y for p in points)

    delta_x = max_x - min_x
    delta_y = max_y - min_y
    delta_max = max(delta_x, delta_y)

    big_triangle = Triangle(
        Point(min_x - 7 * delta_max, min_y - delta_max),
        Point(min_x + delta_max, max_y + 7 * delta_max),
        Point(max_x + 7 * delta_max, min_y - delta_max)
    ) #here 7 is an arbitrary value I chose that can be changed in order to ensure the creation of the triangle

    #print(f"Super-Triangle: {big_triangle}")
    
    triangles = [big_triangle]  # we add the big triangle to the triangles of the triangulation, this will be our starting point

    
    # we re going thorgh each point in order to add it to the triangualtion

    for p in points:

        #print(f"point: {p}") 
        to_eliminate_triangles = [] 

        # First thing we ll identify the triangles that contain the point insdie their circumcircle and elimnate them
        for tri in triangles:
            if tri.contains_point_in_circumcircle(p): 
               to_eliminate_triangles.append(tri) 

       
        # we now create a polygonal hole, meaning we create a polygon using the edges of the traingles we are goinng to elimnate 
        #the edges that are not shared by these triangeles

        hole_edges = []  
        for tri in to_eliminate_triangles:
            for edge in itertools.combinations(tri.vertices, 2): 
             
                #we sort the edges of the trinagles and go through them, if the edge is already in the hole_edges we eliminate, else we add it
                edge = tuple(sorted(edge, key=lambda v: (v.x, v.y)))
                if edge in hole_edges:
                    hole_edges.remove(edge)
                else:
                    hole_edges.append(edge)

        
        for tri in to_eliminate_triangles:
            triangles.remove(tri)

        
        for edge in hole_edges: #we now retriangulate the hole by connecting eacg edge to the current point and adding the triangle to the triangualtin
           if not collinear(edge[0], edge[1], p): 
                new_triangle = Triangle(edge[0], edge[1], p)
                triangles.append(new_triangle)

    triangles = [
        t for t in triangles 
        if not any(v in big_triangle.vertices for v in t.vertices)
    ]

    return triangles  # we finally return the triangles that compose the triangualtion



def plot_triangulation(points, triangles): #dfucntion used to visualzie the triangulation obtained
 
    plt.figure(figsize=(8, 8))

    for p in points:
        plt.plot(p.x, p.y, 'o', color='black')

    for tri in triangles:
        x = [v.x for v in tri.vertices] + [tri.vertices[0].x]
        y = [v.y for v in tri.vertices] + [tri.vertices[0].y]
        plt.plot(x, y, '-', color='blue')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Delaunay Triangulation')
    plt.show()

def count_half_lines(edge_map):
    half_line_count = 0

    for edge, triangle_indices in edge_map.items():
        # A half-line corresponds to an edge shared by only one triangle
        if len(triangle_indices) == 1:
            half_line_count += 1

    return half_line_count


def extend_boundary_edge(circumcenter, p1, p2, p3, length=100):
    #this fucntion is used to represent half-lines, by creating a 'far-point' in the direction in which the line would extend infintely

    mid_x, mid_y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2

    dx, dy = p2.y - p1.y, p1.x - p2.x  # We compute the perpendicular vector of the boundary edge   
    magnitude = (dx**2 + dy**2)**0.5
    dx, dy = dx / magnitude, dy / magnitude

    far_point_x = mid_x + dx * length
    far_point_y = mid_y + dy * length
    far_point = Point(far_point_x, far_point_y)

    p3_test = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x) # here we test on what side of the boundary edge the third point of the triangle is
    #we know that the far point    should be on the opposite side of the line w.r.t. the third vetex
    far_point_test = (p2.x - p1.x) * (far_point.y - p1.y) - (p2.y - p1.y) * (far_point.x - p1.x)
    
    if (p3_test * far_point_test) > 0: # change direction if the far-point and third vertex are on the same side of the line
        dx, dy = -dx, -dy
        far_point_x = mid_x + dx * length
        far_point_y = mid_y + dy * length
        far_point = Point(far_point_x, far_point_y)

    return far_point

def compute_voronoi_from_delaunay(points, delaunay_triangles): # here we compute the voronoi diagram using the delaunay triangulation we have created using our fucntion
   
    circumcenters = []

    for tri in delaunay_triangles:
        circumcenter, _ = tri.compute_circumcenter_and_radius()
        circumcenters.append(circumcenter) #we first compute the circumcenter of each triangle, the circumcenter will be vertexes of the edges of the vornoi diagram

    edge_map = defaultdict(list)
    for i, tri in enumerate(delaunay_triangles):
        for edge in itertools.combinations(tri.vertices, 2):
            edge = tuple(sorted(edge, key=lambda v: (v.x, v.y)))
            edge_map[edge].append(i)  # extract all the edges from the triangulation, to use later
    
    num_half_lines = count_half_lines(edge_map)# the number of half-lines is equalt to the number of boundary edges in the triangualtion, edges that are unique to one triangel
    print(f"Number of half-lines in the Voronoi diagram: {num_half_lines}")


    voronoi_edges = []
    for edge, triangle_indices in edge_map.items():
        if len(triangle_indices) == 2: # if the edge is shared by two triangle, we unite circumcenters of said traingles and crated a voronoi edge
            t1, t2 = triangle_indices
            voronoi_edges.append((circumcenters[t1], circumcenters[t2]))
        elif len(triangle_indices) == 1:  # boundary edge, we use far_pint function to compute the direction in which the half-line/half-edge will extedn
            t1 = triangle_indices[0]
            p1, p2 = edge
            circumcenter = circumcenters[t1]
            for p in delaunay_triangles[t1].vertices:
                if p != p1 and p != p2:
                    p3 = p
                    break
         
            far_point = extend_boundary_edge(circumcenter, p1, p2, p3, length=100)
            voronoi_edges.append((circumcenter, far_point))

    return voronoi_edges

def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def delaunay_to_euclidean_graph(triangles): # for the minimum spanning tree we will use the pro that the eucldiean MST is a subgraph of the delaunay triangulation
    edges = set()
    for triangle in triangles:
        vertices = triangle.vertices
        edges.add((vertices[0], vertices[1]))
        edges.add((vertices[1], vertices[2]))
        edges.add((vertices[2], vertices[0]))

    weighted_edges = []
    for p1, p2 in edges: # to each edge we add it' weight which will be the euclidean distance of the endpoints
        weight = euclidean_distance(p1, p2)
        weighted_edges.append((p1, p2, weight))
    return weighted_edges

def kruskal_algorithm(points, edges):
    #simple implementaion of the kruskal's algorithm in whcih we add the minimum edge that doesn't form a cycle until no more
    parent = {point: point for point in points}

    def find(point):
        if parent[point] != point:
            parent[point] = find(parent[point])
        return parent[point]

    def union(p1, p2):
        root1 = find(p1)
        root2 = find(p2)
        if root1 != root2:
            parent[root2] = root1

    mst = []
    edges.sort(key=lambda edge: edge[2]) 

    for p1, p2, weight in edges:
        if find(p1) != find(p2):
            union(p1, p2)
            mst.append((p1, p2, weight))

    return mst

# the following funcrion are stricly used to visualzie the reults of the algorithm in a plot 
def plot_mst(points, mst):
    plt.figure(figsize=(8, 8))

    for p in points:
        plt.plot(p.x, p.y, 'o', color='black')

    for p1, p2, _ in mst:
        plt.plot([p1.x, p2.x], [p1.y, p2.y], '-', color='green')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Minimum Spanning Tree')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def plot_triangulation_and_voronoi(points, triangles, voronoi_edges):
    plt.figure(figsize=(8, 8))

    for p in points:
        plt.plot(p.x, p.y, 'o', color='black')

    for tri in triangles:
        x = [v.x for v in tri.vertices] + [tri.vertices[0].x]
        y = [v.y for v in tri.vertices] + [tri.vertices[0].y]
        plt.plot(x, y, '-', color='blue', label='Delaunay Triangulation')

    for edge in voronoi_edges:
        x = [edge[0].x, edge[1].x]
        y = [edge[0].y, edge[1].y]
        plt.plot(x, y, '-', color='red', label='Voronoi Diagram')

    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    
    margin = 1.5 * max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
    plt.xlim(min(x_coords) - margin, max(x_coords) + margin)
    plt.ylim(min(y_coords) - margin, max(y_coords) + margin)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Delaunay Triangulation and Voronoi Diagram')
    plt.legend(['Delaunay Triangulation', 'Voronoi Diagram'], loc='upper right')
    plt.show()

def mst_length_for_lambda(lambda_val): 
    points = [
        Point(-1, 6),
        Point(-1, -1),
        Point(4, 7),
        Point(6, 7),
        Point(1, -1),
        Point(-5, 3),
        Point(-2, 3),
        Point(2 - lambda_val, 3)
    ]
    triangulation = delaunay_triangulation(points)
    edges = delaunay_to_euclidean_graph(triangulation)
    mst = kruskal_algorithm(points, edges)
    
    mst_length = sum(weight for _, _, weight in mst)
    return mst_length


def exercise_1():
    #we simply solve this exercise by computing the triangulation and voronoi diagram using the algorithms above
    points_ex1 = [Point(3, -5), Point(-6, 6), Point(6, -4), Point(5, -5), Point(9, 10)]
    triangulation_ex1 = delaunay_triangulation(points_ex1)

    num_edges_ex1 = len(
        set(tuple(sorted(edge, key=lambda v: (v.x, v.y))) 
            for t in triangulation_ex1 
            for edge in combinations(t.vertices, 2))
    )

    print("Exercise 1:")
    print("Number of triangles in the triangulation:", len(triangulation_ex1))
    print("Number of edges:", num_edges_ex1)

    voronoi_edges_ex1 = compute_voronoi_from_delaunay(points_ex1, triangulation_ex1)
    plot_triangulation_and_voronoi(points_ex1, triangulation_ex1, voronoi_edges_ex1)

def exercise_2():
    #we have already added the points P7 and P8, the explination for choosing them will be in the pdf file associated to this homework
    points_ex2 = [
        Point(5, -1), Point(7, -1), Point(9, -1), Point(7, -3),
        Point(11, -1), Point(-9, 3), Point(24, -4), Point(10, 10)
    ]
    triangulation_ex2 = delaunay_triangulation(points_ex2)
    voronoi_ex2 = compute_voronoi_from_delaunay(points_ex2, triangulation_ex2)

    print("Exercise 2:")
    plot_triangulation_and_voronoi(points_ex2, [], voronoi_ex2)

def exercise_3():

    #to solve this exercise we test several values for lambda between the interval -8 and 7, better explained in geogebra why this particualr interval
    #we afterards try to see if there exists such a lambda vlaue for which the leght is smaller than when Q would coincide with F or P
    best_lambda = None
    min_mst_length = float('inf')

    lambda_values = np.linspace(-8, 7, num=50000)

    for lambda_val in lambda_values:
        length = mst_length_for_lambda(lambda_val)
        if length < min_mst_length:
            min_mst_length = length
            best_lambda = lambda_val

    print(f"MST length for lambda=4: {mst_length_for_lambda(4)}")
    print(f"MST length for lambda=7: {mst_length_for_lambda(7)}")
    print(f"Best lambda: {best_lambda}, Minimum MST length: {min_mst_length}")

    best_points = [
        Point(-1, 6), Point(-1, -1), Point(4, 7), Point(6, 7),
        Point(1, -1), Point(-5, 3), Point(-2, 3), Point(2 - best_lambda, 3)
    ]

    best_triangulation = delaunay_triangulation(best_points)
    best_edges = delaunay_to_euclidean_graph(best_triangulation)
    best_mst = kruskal_algorithm(best_points, best_edges)

    plot_triangulation(best_points, best_triangulation)
    plot_mst(best_points, best_mst)

def exercise_4():
    #Simply drawing the voronoi diagram for this set of points
    M = [
        Point(1, -1), Point(0, 0), Point(-1, 1), Point(-2, 2), Point(-3, 3), Point(-4, 4),
        Point(0, 0), Point(1, -1), Point(2, -2), Point(3, -3), Point(4, -4), Point(5, -5),
        Point(0, 0), Point(0, 1), Point(0, 2), Point(0, 3), Point(0, 4), Point(0, 5)
    ]

    triangulation_ex4 = delaunay_triangulation(M)
    voronoi_ex4 = compute_voronoi_from_delaunay(M, triangulation_ex4)

    print("Exercise 4:")
    plot_triangulation_and_voronoi(M, triangulation_ex4, voronoi_ex4)


def exercise_5():
    # here we use the points from exercise 1 as the first set
    # the second set was computed from the idea of a triangulation in which the we have a triangle and a forth point that is inside the triangle, for that we used 
    # the circumcenter of the triangle
    point_m1 = [Point(3, -5), Point(-6, 6), Point(6, -4), Point(5, -5), Point(9, 10)]
    points_m2 = [Point(-7, 0), Point(0, 5), Point(3, -7), Point(-0.62, -1.53)]

    triangulation_m1 = delaunay_triangulation(point_m1)
    triangulation_m2 = delaunay_triangulation(points_m2)

    num_edges_m1 = len(
        set(tuple(sorted(edge, key=lambda v: (v.x, v.y))) 
            for t in triangulation_m1 
            for edge in combinations(t.vertices, 2))
    )

    num_edges_m2 = len(
        set(tuple(sorted(edge, key=lambda v: (v.x, v.y))) 
            for t in triangulation_m2 
            for edge in combinations(t.vertices, 2))
    )

    print("Exercise 5:")

    print("M!:")
    print("Number of triangles in the triangulation:", len(triangulation_m1)) 
    print("Number of edges:", num_edges_m1)

    print("M2:")
    print("Number of triangles in the triangulation:", len(triangulation_m2)) 
    print("Number of edges:", num_edges_m2)

    if(len((triangulation_m1)) == len(triangulation_m2) and len(point_m1)!= len(points_m2)):
        print("We have M1 and M2, sets of points having differnt cardinality, but which admit triangualtion with exactly three faces")

    voronoi_m1 = compute_voronoi_from_delaunay(point_m1, triangulation_m1)
    voronoi_m2 = compute_voronoi_from_delaunay(points_m2, triangulation_m2)

    plot_triangulation_and_voronoi(point_m1, triangulation_m1, voronoi_m1)
    plot_triangulation_and_voronoi(points_m2, triangulation_m2, voronoi_m2)


def exercise_6():
    #we simply compute the triangulation based on a given value for the M point and print the number of faces and the number of edges
    X = float(input("Enter lambda for point M, lambda should be real:"))
    points_ex6 = [Point(1, 1), Point(1, -1), Point(-1, -1), Point(-1, 1), Point(0, -2), Point(0, X)]

    triangulation_ex6 = delaunay_triangulation(points_ex6)
    num_edges_ex6 = len(
        set(tuple(sorted(edge, key=lambda v: (v.x, v.y))) 
            for t in triangulation_ex6 
            for edge in combinations(t.vertices, 2))
    )

    print("Exercise 6:")
    print("Number of triangles in the triangulation:", len(triangulation_ex6))
    print("Number of edges:", num_edges_ex6)

    plot_triangulation(points_ex6, triangulation_ex6)


def main():
    exercises = {
        1: exercise_1,
        2: exercise_2,
        3: exercise_3,
        4: exercise_4,
        5: exercise_5,
        6: exercise_6,
    }

    print("Available exercises: 1, 2, 3, 4, 5, 6")
    choice = input("Enter the exercise number to run: ").strip()

    try:
        if choice.isdigit():
            choice = int(choice)
        if choice in exercises:
            exercises[choice]()
        else:
            print(f"Invalid choice: {choice}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()