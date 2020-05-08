#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
using namespace std;
map <string,float> scanGain(){
    ifstream file;
    string line,tmp;
    file.open("src/gain.csv");
    vector <string> tmp_info;
    vector <float> tmp_val;
    map <string,float> gain_map;
    getline(file,line);
    {
        stringstream str(line);
        while(getline(str,tmp,',')){
            tmp_info.push_back(tmp);
        }
    }
    getline(file,line);
    {
    stringstream str(line);
    while(getline(str,tmp,',')){
            stringstream str_num(tmp);
            float num;
            str_num>>num;
            tmp_val.push_back(num);
        }
        file.close();
        for(int i =0;i<tmp_info.size();i++){
          gain_map[tmp_info[i]]=tmp_val[i];
        }
    }
    return gain_map;
}
vector <string> scanAttr(){
    ifstream file;
    file.open("../analyze_dataset/tab.csv");
    string line,tmp;
    vector <string> tmp_a,attr;
    getline(file,line); 
    getline(file,line);
    file.close();
    stringstream str(line);
    while(getline(str,tmp,',')){
        tmp_a.push_back(tmp);
    }
    for(int i=2;i<tmp_a.size()-2;i++){
        attr.push_back(tmp_a[i]);
    }
    return attr;
}

vector <vector<float>> corr_matrix(){
    ifstream file;
    file.open("src/corrMatrix.csv");
    string line,tmp;
    vector <vector<float>> matr;
    while(getline(file,line)){
        stringstream str(line);
        vector <float >tmp_m;
        while(getline(str,tmp,',')){
            stringstream ss(tmp);
            float num;
            ss>>num;
            tmp_m.push_back(num);
        }
       matr.push_back(tmp_m); 
    }
    file.close();
    return matr;
}

int main(){
    vector <string> attr=scanAttr();
    vector<string> broke_attr,choose_attr;
    vector<vector<float>> matr=corr_matrix();
    map<string,float> gain_map=scanGain();
    for(int i=0;i<matr.size();i++){
        for(int j=i+1;j<matr[0].size();j++){
            if(fabs(matr[i][j])>0.8){
                if(find(broke_attr.begin(),broke_attr.end(),attr[j])==broke_attr.end()){
                if(gain_map[attr[i]]>gain_map[attr[j]]){
                    broke_attr.push_back(attr[j]);
                }
                else{
                    broke_attr.push_back(attr[i]);
                }
                }
            } 
        }
    }
    for(auto it:gain_map){
        if(it.second<2){
            if(find(broke_attr.begin(),broke_attr.end(),it.first)==broke_attr.end())
                broke_attr.push_back(it.first);
        }
    }
    for(auto it:broke_attr){
        cout<<it<<",";
     }
    cout<<broke_attr.size()<<endl;
    for(auto it :gain_map){
        if(find(broke_attr.begin(),broke_attr.end(),it.first)==broke_attr.end()){
            choose_attr.push_back(it.first);
            cout<<it.first<<",";
        }
    }
    cout<<endl<<choose_attr.size();
}
