#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <cmath>
#include <cassert>

double dist(std::vector<double> &point_1, std::vector<double> &point_2)
{
    return sqrt(pow(point_1[0]-point_2[0],2) + pow(point_1[1] - point_2[1],2));
}

double stress(std::vector<std::vector<int>> &data , std::vector<std::vector<double>> &layout)
{
    double stress = 0;
    for(auto edge : data)
    {
        stress += pow(dist(layout[edge[0]],layout[edge[1]]) - edge[2],2);
    }
    return stress;
}


void lay(std::vector<std::vector<int>> &data, std::vector<std::vector<double>> &layout, std::vector<double> schedule)
{
    auto r_engine = std::default_random_engine {};
    for(auto eta : schedule)
    {   
        std::shuffle(data.begin(), data.end(), r_engine);

        for(auto edge : data)
        {
            int i = edge[0], j = edge[1], d_ij = edge[2];
            assert( i != j );
            double r = (dist(layout[i],layout[j]) - d_ij)/2/dist(layout[i],layout[j]);
            double x = layout[i][0] - layout[j][0], y = layout[i][1] - layout[j][1];
            layout[i][0] -= x*r*eta/pow(d_ij,2);
            layout[i][1] -= y*r*eta/pow(d_ij,2);
            layout[j][0] += x*r*eta/pow(d_ij,2);
            layout[j][1] += y*r*eta/pow(d_ij,2);
            if (std::isnan(r) || std::isnan(layout[i][0]) || std::isnan(layout[j][0]))
            {
                std::cerr << "-------------------------------------------------\n";
                std::cerr << layout[i][0] << " , " << layout[i][1] << std::endl;
                std::cerr << layout[j][0] << " , " << layout[j][1] << std::endl;
                std::cerr << d_ij << std::endl;
                std::cerr << r << std::endl;
                std::cerr << x << std::endl;
                std::cerr << eta << std::endl;
                std::cerr << dist(layout[i],layout[j]);
                std::cerr << "-------------------------------------------------\n";
                break;
            }
        }
        std::cerr << "The Stress is: " << stress(data,layout) << '\n';

    }
}

int main()
{
    /*
        In this simple implementation the input is in the format of a file containing the graph edges.
        The first line describes the input it contains the number of nodes and the number of lines following it.
        The rest of the lines have the format 
            i j d(i,j)
        i is the number of the first node, j is the number of another node, and d(i,j) is the distance between the two nodes.
        the calculation of the distances is performed using dijkstra's algorithm. I decided to take the input with precalcualted
        path lengths since the bottleneck in the layout algorithm is the SGD, the calculation of the shortests paths can also be parallelized
        but that's not the focus rn.
    */
    std::ifstream graph ("input.mtx");
    std::vector<std::vector<int>> data;
    std::vector<std::vector<double>> layout;
    long nodes, edges;
    
    graph >> nodes;
    graph >> edges;
    for(int k = 0; k < edges; k++) 
    {
        std::vector<int> edge;
        int i, j, d_ij;
        graph >> i >> j >> d_ij;
        edge.push_back(i);
        edge.push_back(j);
        edge.push_back(d_ij);
        data.push_back(edge);
    }

    for(int i = 0; i < nodes; i++)
    {
        std::vector<double> point;
        point.push_back(rand()/10000.);
        point.push_back(rand()/10000.);
        layout.push_back(point);
    }
    // for(int i = 0; i < nodes ; i++)
    // {
    //     std::cout << layout[i][0] << " " << layout[i][1] << '\n';
    // }

    // std::cout << "\nEdges , Nodes\n";
    // std::cout << nodes << " , " << edges << '\n';
    // std::cout << layout.size() << " , " << data.size() << '\n'; 

    std::vector<double> annealing_sch;
    for(int i = 0 ; i < 20000; i++)
    {
        annealing_sch.push_back(2);
    }
    for(int i = 0 ; i < 20000; i++)
    {
        annealing_sch.push_back(1);
    }
    for(int i = 0 ; i < 12000; i++)
    {
        annealing_sch.push_back(.1);
    }
    lay(data,layout,annealing_sch);
    for(int i = 0; i < nodes ; i++)
    {
        std::cout << layout[i][0] << " " << layout[i][1] << '\n';
    }
}
