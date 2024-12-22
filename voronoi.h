//
// Created by rares655 on 12/21/24.
//

#ifndef VORONOI_H
#define VORONOI_H
#include <queue>
#include <utility>

using namespace std;

inline double boundX1 = 0, boundX2 = 0;
inline double boundY1 = 0, boundY2 = 0;
//these are the coordinates for the bounding box of the diagram, soley used for visualization purposes

typedef pair<double, double> VoronoiPoint;
#define x first
#define y second // to represent the point/site of the voronoi diagram we use the pair data type from the std library

struct EdgeData;
struct Parabola_Data; // we define the edge and arc data structures that we will use later

inline vector<EdgeData*> VoronoiEdges;  // vector that will hold all the voronoi edges, this will be the final result of the algorithm

struct FortuneEvent {  // data structure used to hold informations about the events (point event/ vertex event)
    double x; //coordonate of the event, that we use to order them in the priority queue, in order to simulate the sweeping line
    VoronoiPoint p;
    Parabola_Data *a;
    bool Okay;

    FortuneEvent(double xx, VoronoiPoint pp, Parabola_Data *aa)
       : x(xx), p(std::move(pp)), a(aa), Okay(true) {}
};

struct Parabola_Data { // the arc data structure representd the arcs that appear in the beachline that end up forming a Voronoi Edge
    VoronoiPoint p;
    Parabola_Data *previous_Arc, *next_Arc;
    EdgeData *start_point, *end_point; // the edge assoicated with the arc, being described by start and ed points
    FortuneEvent *e;

    explicit Parabola_Data(VoronoiPoint pp, Parabola_Data *a=nullptr, Parabola_Data *b=nullptr)
     : p(std::move(pp)), previous_Arc(a), next_Arc(b), start_point(nullptr), end_point(nullptr), e(nullptr) {}
};

struct EdgeData {
    VoronoiPoint start_point, finish_point;
    bool finished_edge; //we represent edges by theier starting and ending point, and also having a bool value that tells us if the edges is finished
    //we will use these in the case of half edges/ half lines, in order to find a 'false end point' for visualization purposes

    explicit EdgeData(VoronoiPoint p)
       : start_point(std::move(p)), finish_point(0,0), finished_edge(false)
    { VoronoiEdges.push_back(this); }

    void finish_edge(const VoronoiPoint &p) {
        if (finished_edge)
            return;
        finish_point = p;
        finished_edge = true;
    }
};

struct comparison_structure { // in order to compare points and events in the prioirty queue we use this strcutre
    bool operator()(const VoronoiPoint &point1, const VoronoiPoint &point2) const {return point1.x==point2.x ? point1.y>point2.y : point1.x>point2.x;}
    bool operator()(const FortuneEvent *event1, const FortuneEvent *event2) const {return event1->x>event2->x;}
};

inline priority_queue<VoronoiPoint,  vector<VoronoiPoint>,  comparison_structure> voronoi_sites;
inline priority_queue<FortuneEvent*, vector<FortuneEvent*>, comparison_structure> fortune_events;

inline Parabola_Data *root = nullptr;
void insert_point(const VoronoiPoint &insertion_point);
void voronoi_site_handler();
void voronoi_event_handler();
void validate_circle_event(Parabola_Data *parabola, double x0);
bool check_intersection(const VoronoiPoint &site, const Parabola_Data *parabola, VoronoiPoint *result = nullptr);
bool circle(const VoronoiPoint &Vertex1, const VoronoiPoint &Vertex2, const VoronoiPoint &Vertex3, double *x, VoronoiPoint *CircleCenter);
VoronoiPoint find_intersection(const VoronoiPoint &Site1, const VoronoiPoint &Site2, double sweep_line);
void end_edges();
void print_voronoi_edges();

#endif //VORONOI_H
