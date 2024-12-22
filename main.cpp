#include "voronoi.h"
#include <set>
#include <cmath>
#include <iostream>

int main()
{
   std::set<VoronoiPoint> unique_points; // in order to not affect the voronoi diagram, when reading the input
   //if two points coincide, then we add it only once, as to avoid anomalies, we use the se to check uniqueness
   int number_of_points;
   std::cin >> number_of_points;

   for (int i = 0; i < number_of_points; i++) {
      VoronoiPoint VoronoiSite;
      std::cin >> VoronoiSite.x >> VoronoiSite.y;

      // we add the point read only if it's not already in the set, i.e. if it unique
      if (unique_points.insert(VoronoiSite).second) {
         voronoi_sites.push(VoronoiSite);

         if (VoronoiSite.x < boundX1)
            boundX1 = VoronoiSite.x;
         if (VoronoiSite.y < boundY1)
            boundY1 = VoronoiSite.y;
         if (VoronoiSite.x > boundX2)
            boundX2 = VoronoiSite.x;
         if (VoronoiSite.y > boundY2)
            boundY2 = VoronoiSite.y;
      }
   }
   const double dx = (boundX2-boundX1+1)/5.0;
   const double dy = (boundY2-boundY1+1)/5.0;
   boundX1 -= dx;
   boundX2 += dx;
   boundY1 -= dy;
   boundY2 += dy;

   while (!voronoi_sites.empty())
      if (!fortune_events.empty() && fortune_events.top()->x <= voronoi_sites.top().x)
         voronoi_event_handler();
      else
         voronoi_site_handler();

   while (!fortune_events.empty())// we keep on handling the events until there are no more events, meaning that we have reached the end of the algoithm
      voronoi_event_handler();

   end_edges();  // we finish the unended edges, the half-lines for visualization purpooses
   print_voronoi_edges();
}
void print_voronoi_edges()
{
   for (const auto & VoronoiEdge : VoronoiEdges) {
      VoronoiPoint start_edge = VoronoiEdge->start_point;
      VoronoiPoint end_edge = VoronoiEdge->finish_point;
      //here we do the printing of these edges in this manner in order to be able to copy the edge list and put in an python script for visualziaton
      cout <<"((" <<start_edge.x << ", " << start_edge.y << "), (" << end_edge.x << ", " << end_edge.y <<")),\n";
   }
}
void voronoi_site_handler()
{

   // we get the point from the priorty qquue and generate an according parablic arc for it
   VoronoiPoint p = voronoi_sites.top();
   voronoi_sites.pop();
   insert_point(p);
}
void insert_point(const VoronoiPoint &insertion_point)
{
   if (!root) {
      root = new Parabola_Data(insertion_point);
      return;
   }

   for (Parabola_Data *existent = root; existent; existent = existent->next_Arc) {
      VoronoiPoint aux_point1;
      VoronoiPoint aux_point2;
      if (check_intersection(insertion_point,existent,&aux_point1)) {
         // check intersection of the new parabola with parabola existent, in order to connect them in the beachline
         if (existent->next_Arc && !check_intersection(insertion_point,existent->next_Arc, &aux_point2)) {
            existent->next_Arc->previous_Arc = new Parabola_Data(existent->p,existent,existent->next_Arc);
            existent->next_Arc = existent->next_Arc->previous_Arc;
         }
         else existent->next_Arc = new Parabola_Data(existent->p,existent);
         existent->next_Arc->end_point = existent->end_point;

         // we add the point to be inserted in between the two parabolas
         existent->next_Arc->previous_Arc = new Parabola_Data(insertion_point,existent,existent->next_Arc);
         existent->next_Arc = existent->next_Arc->previous_Arc;
         existent = existent->next_Arc;

         // we add new hald edges to the ends of the already existent parabola dn finally check if any new circle events were formed
         existent->previous_Arc->end_point = existent->start_point = new EdgeData(aux_point1);
         existent->next_Arc->start_point = existent->end_point = new EdgeData(aux_point1);
         validate_circle_event(existent, insertion_point.x);
         validate_circle_event(existent->previous_Arc, insertion_point.x);
         validate_circle_event(existent->next_Arc, insertion_point.x);
         return;
      }
   }

   // Special case: If p never intersects an arc, append it to the list.
   Parabola_Data *i;
   for (i = root; i->next_Arc; i=i->next_Arc) {}

   i->next_Arc = new Parabola_Data(insertion_point,i);
   // Insert segment between p and i
   VoronoiPoint start;
   start.x = boundX1;
   start.y = (i->next_Arc->p.y + i->p.y) / 2;
   i->end_point = i->next_Arc->start_point = new EdgeData(start);
}
void end_edges()
{
   const double l = boundX2 + (boundX2-boundX1) + (boundY2-boundY1);
   for (const Parabola_Data *parabola = root; parabola->next_Arc; parabola = parabola->next_Arc)
      if (parabola->end_point)
         parabola->end_point->finish_edge(find_intersection(parabola->p, parabola->next_Arc->p, l*2));
}
void voronoi_event_handler()
{
   const FortuneEvent *e = fortune_events.top();
   fortune_events.pop();
   // we get the vent from the queue, check validation and proceed with handling it
   if (e->Okay) {
      // we create a new edge starting at the event's site.
      auto *s = new EdgeData(e->p);

      const Parabola_Data *parabola = e->a;

      // we update the beachline of parabolas as we eliminate the parabol assoicated to this circle event
      if (parabola->previous_Arc) {
         parabola->previous_Arc->next_Arc = parabola->next_Arc;
         parabola->previous_Arc->end_point = s;
      }
      if (parabola->next_Arc) {
         parabola->next_Arc->previous_Arc = parabola->previous_Arc;
         parabola->next_Arc->start_point = s;
      }

      // we finally finish the edge genrated by this parabola and circle/ vertex event
      if (parabola->start_point)
         parabola->start_point->finish_edge(e->p);
      if (parabola->end_point)
         parabola->end_point->finish_edge(e->p);

      if (parabola->previous_Arc)
         validate_circle_event(parabola->previous_Arc, e->x);
      if (parabola->next_Arc)
         validate_circle_event(parabola->next_Arc, e->x);
   }
   delete e;
}
void validate_circle_event(Parabola_Data *parabola, const double x0)
{
   // we invalidate any other old event and look for new possible circle events that can occur during the sweeping of the line
   if (parabola->e && parabola->e->x != x0)
      parabola->e->Okay = false;
   parabola->e = nullptr;
   if (!parabola->previous_Arc || !parabola->next_Arc)
      return;

   double x;
   VoronoiPoint o;

   if (circle(parabola->previous_Arc->p, parabola->p, parabola->next_Arc->p, &x,&o) && x > x0) {
      parabola->e = new FortuneEvent(x, o, parabola);
      fortune_events.push(parabola->e);
   }
}
bool circle(const VoronoiPoint &Vertex1, const VoronoiPoint &Vertex2, const VoronoiPoint &Vertex3, double *x, VoronoiPoint *CircleCenter)
{

   if ((Vertex2.x-Vertex1.x)*(Vertex3.y-Vertex1.y) - (Vertex3.x-Vertex1.x)*(Vertex2.y-Vertex1.y) > 0)
      return false;

   // we define the circle made by the three vertices using the general equation of a conic
   const double A = Vertex2.x - Vertex1.x;
   const double B = Vertex2.y - Vertex1.y;
   const double C = Vertex3.x - Vertex1.x;
   const double D = Vertex3.y - Vertex1.y;
   const double E = A*(Vertex1.x+Vertex2.x) + B*(Vertex1.y+Vertex2.y);
   const double F = C*(Vertex1.x+Vertex3.x) + D*(Vertex1.y+Vertex3.y);
   const double G = 2*(A*(Vertex3.y-Vertex2.y) - B*(Vertex3.x-Vertex2.x));
   if (G == 0) return false; //collinear case for points, we ignore the handling of these, in order to avoid division by zero

   CircleCenter->x = (D*E-B*F)/G;
   CircleCenter->y = (A*F-C*E)/G;

   *x = CircleCenter->x + sqrt( pow(Vertex1.x - CircleCenter->x, 2) + pow(Vertex1.y - CircleCenter->y, 2) );
   return true;
}
bool check_intersection(const VoronoiPoint &site, const Parabola_Data *parabola, VoronoiPoint *result)
{
   if (parabola->p.x == site.x) return false;

   double a = 0,b = 0;
   if (parabola->previous_Arc)
      a = find_intersection(parabola->previous_Arc->p, parabola->p, site.x).y;
   if (parabola->next_Arc)
      b = find_intersection(parabola->p, parabola->next_Arc->p, site.x).y;
   if ((!parabola->previous_Arc || a <= site.y) && (!parabola->next_Arc || site.y <= b)) {
      result->y = site.y;
      result->x = (parabola->p.x*parabola->p.x + (parabola->p.y-result->y)*(parabola->p.y-result->y) - site.x*site.x)/ (2*parabola->p.x - 2*site.x);
      return true;
   }
   return false;
}
// we find the intersection point between two parabolas defined by Site1 and Site2, given the current sweep line position l.
VoronoiPoint find_intersection(const VoronoiPoint &Site1, const VoronoiPoint &Site2, const double sweep_line)
{
    VoronoiPoint IntersectionPoint, aux_site = Site1;

    // if the two sites have the same x-coordinate (vertical alignment), then their intersecton is the mid-point between their y's

    if (Site1.x == Site2.x)
        IntersectionPoint.y = (Site1.y + Site2.y) / 2;
    else if (Site2.x == sweep_line)  // Site2 is directly on the sweep line (l) the intersection lies directly at Site2's y-coordinate.
        IntersectionPoint.y = Site2.y;
    else if (Site1.x == sweep_line) {// Site1 is directly on the sweep line (l) the intersection lies directly at Site1's y-coordinate.
        IntersectionPoint.y = Site1.y;
        aux_site = Site2;
    } else {  // we use the quadratic formula to calculate the intersection's y-coordinate.
        const double distance_1 = 2 * (Site1.x - sweep_line);
        const double distance_2 = 2 * (Site2.x - sweep_line);

        // we compute the coefficients that characterize the parabola
        const double parabola_a = 1 / distance_1 - 1 / distance_2;
        const double parabola_b = -2 * (Site1.y / distance_1 - Site2.y / distance_2);
        const double parabola_c = (Site1.y * Site1.y + Site1.x * Site1.x - sweep_line * sweep_line) / distance_1 - (Site2.y * Site2.y + Site2.x * Site2.x - sweep_line * sweep_line) / distance_2;

        // we finally solve the quadratic euqation to find solution for y coordonate
        IntersectionPoint.y = (-parabola_b - sqrt(parabola_b * parabola_b - 4 * parabola_a * parabola_c)) / (2 * parabola_a);
    }

    // we compute the x-coordinate of the intersection point using the parabola equation.
    IntersectionPoint.x = (aux_site.x * aux_site.x +(aux_site.y - IntersectionPoint.y) * (aux_site.y - IntersectionPoint.y)- sweep_line * sweep_line) / (2 * aux_site.x - 2 * sweep_line);
    return IntersectionPoint;
}
