// Copyright (c) 2019, University of Washington All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// 
// Redistributions of source code must retain the above copyright notice, this list
// of conditions and the following disclaimer.
// 
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// 
// Neither the name of the copyright holder nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "layout.h"
using namespace std;

#define ALLOC_NAME "default_allocator"

vector<int> node_lengths;
vector<int> node_ids;
vector<int> positions;

path_data_t path_data;
node_data_t node_data;


void get_data(){
    // cout<<"here";
    fstream file1,file2,file3;
    string word, filename1, filename2, filename3;
    filename1 = "node_length.txt";
 
    file1.open(filename1.c_str());

    while (file1 >> word)
    {
        node_lengths.push_back(stoi(word)); 
    }

    filename2 = "vis_id.txt";
    file1.close();
    file2.open(filename2.c_str());

    while (file2 >> word)
    {
        // cout<<word<<endl;
        node_ids.push_back(stoi(word)); 
    }

    filename3 = "pos.txt";
    file2.close();
    file3.open(filename3.c_str());

    while (file3 >> word)
    {
        positions.push_back(stoi(word)); 
    }
    file3.close();

    //Node Data
    node_data.node_count=node_lengths.size();
    for (int node_idx = 0; node_idx < node_lengths.size(); node_idx++) {
        node_t n_tmp;
        n_tmp.seq_length = node_lengths[node_idx];

        n_tmp.coords[0] = float(node_idx*2);
        n_tmp.coords[1] = float(node_idx * 2);
        n_tmp.coords[2] = float(node_idx*2 + 1);
        n_tmp.coords[3] = float(node_idx * 2 + 1);

        node_data.nodes.push_back(n_tmp);
    }
    
    //Path Data
    path_data.path_count = 0;
    path_data.total_path_steps = 0;
    uint64_t step_id=0;
    uint64_t i=0;
    uint32_t pid=0;
    int64_t pos=0;
    uint64_t array_num=0;
    uint64_t array_length=0;
    while(i<positions.size()){
        array_length=positions[i]*2;
        for(uint64_t j=0;j<positions[i]*2;j+=2){
            for(uint64_t k=j+2; k<positions[i]*2;k+=2){
                pos=1;
                path_data.path_count++;
                path_data.total_path_steps+=(k-j)/2+1;
                path_t p_temp;
                p_temp.step_count=(k-j)/2+1;
                p_temp.first_step_in_path=step_id;
                // step_id+=(k-j)/2;
                for(uint64_t z=j; z<=k; z+=2){
                    //cout<<array_num*array_length+z<<endl;
                    path_element_t e_temp;
                    e_temp.pidx=pid;
                    e_temp.node_id=node_ids[array_num*array_length+z]/2;
                    e_temp.pos=pos;
                    p_temp.elements.push_back(e_temp);
                    //if(z!=k){
                        path_data.element_array.push_back(e_temp);
                        step_id++;
                    //}
                    pos+=node_data.nodes[e_temp.node_id].seq_length;
                }
                path_data.paths.push_back(p_temp);
                pid++;
            }
        }
        i=i+positions[i]*2+1;
        array_num++;
    }
    
    // cout<<path_data.total_path_steps<<endl;
    // cout<<step_id<<endl;
}

void sgd_layout(){
    double eta_max = 17000*17000;
    double iter_max = 20;
    double lambda = log(17000*17000 / 0.1) / ((double) iter_max - 1);
    // initialize step sizes
    std::vector<double> etas;
    std::vector<int> nodes_used(15);
    etas.reserve(iter_max+1);
    for (int64_t t = 0; t <= iter_max; t++) {
        etas.push_back(eta_max * exp(-lambda * (abs(t))));
    }

    for(int64_t e_id=0; e_id<etas.size(); e_id++){
        double eta=etas[e_id];
        if(e_id%200==0) cout<<e_id<<endl;
        for (int i = 0; i<iter_max; i++ ){
    uint32_t step_idx = uint32_t(floor((double(rand())/(RAND_MAX)) * float(path_data.total_path_steps)));

    // cout<<step_idx<<endl;
    
    uint32_t path_idx = path_data.element_array[step_idx].pidx;
    path_t p = path_data.paths[path_idx];
    if(p.step_count<2) continue;
    uint32_t s1_idx = uint32_t(rand()%p.step_count);
    uint32_t s2_idx;
    
        do {
            s2_idx = uint32_t(rand()%p.step_count);
        } while (s1_idx == s2_idx);


     uint32_t n1_id = p.elements[s1_idx].node_id;

     uint32_t n2_id = p.elements[s2_idx].node_id;
    nodes_used[n1_id]=1;
    nodes_used[n2_id]=1;
    
    double term_dist = std::abs(static_cast<double> (p.elements[s1_idx].pos) - static_cast<double> (p.elements[s2_idx].pos));

    if (term_dist < 1e-9) {
        term_dist = 1e-9;
    }

    double w_ij = 1.0 / term_dist;

    double mu = eta * w_ij;
    if (mu > 1.0) {
        mu = 1.0;
    }

    double d_ij = term_dist;

    float *x1 = &node_data.nodes[n1_id].coords[0];
    float *x2 = &node_data.nodes[n2_id].coords[0];
    float *y1 = &node_data.nodes[n1_id].coords[1];
    float *y2 = &node_data.nodes[n2_id].coords[1];
    double x1_val = double(*x1);
    double x2_val = double(*x2);
    double y1_val = double(*y1);
    double y2_val = double(*y2);

    double dx = x1_val - x2_val;
    double dy = y1_val - y2_val;

    if (dx == 0.0) {
        dx = 1e-9;
    }

    double mag = sqrt(dx * dx + dy * dy);
    double delta = mu * (mag - d_ij) / 2.0;

    double r = delta / mag;
    double r_x = r * dx;
    double r_y = r * dy;
    *x1=x1_val-r_x;
    *x2=x2_val+r_x;
    *y1=y1_val-r_y;
    *y2=y2_val+r_y;
}
}
    for(int n=0; n<node_lengths.size(); n++){
        cout<<n<<" "<<node_data.nodes[n].seq_length<<" "<<node_data.nodes[n].coords[0]<<" "<<node_data.nodes[n].coords[1]<<endl;
    }
    // for(int n=0; n<path_data.element_array.size(); n++){
    //     nodes_used[path_data.element_array[n].node_id]=1;
    // }
    // for(int n=0; n<nodes_used.size(); n++){
    //     cout<<nodes_used[n]<<endl;
    // }
}

int main(){
    get_data();
    sgd_layout();
    ofstream outfile;
    string outp;
    outp="sgd_out.txt";

    outfile.open(outp.c_str());
    for(int n=0; n<node_lengths.size(); n++){
        outfile<<n<<" "<<node_data.nodes[n].seq_length<<" "<<node_data.nodes[n].coords[0]<<" "<<node_data.nodes[n].coords[1]<<endl;
    }
    outfile.close();
}
