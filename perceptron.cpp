#include <iostream>
#include <map>
#include <vector>
#include <sstream>
#include <fstream>
#include <cmath>
using namespace std;
vector <vector<vector<float>>> createModel(vector<vector <float>> & bias,int number_in,vector<int> &number_nodes,float init_val){
    vector <vector<vector<float>>> weight_matr;
    vector <vector <float>> first_density;
    vector <float> first_b(number_nodes[0],init_val);
    for(int i=0;i<number_nodes[0];i++){
        vector <float> tmp;
        // инициализация весовых коэффицентов для нейронов первого скрытого слоя
        for(int f=0;f<number_in;f++){
            tmp.push_back(powf(-1,f)*0.03*(f+1)*(i+1));
        }
        first_density.push_back(tmp);
    }
    weight_matr.push_back(first_density);
    bias.push_back(first_b);
    for(int i=1;i<number_nodes.size();i++){
        vector <float> bias_dens(number_nodes[i],init_val);
        vector<vector<float>> density;
        // инициализация весовых коэффицентов для 2 ... n слоев
        for(int j=0;j<number_nodes[i];j++){    
            vector <float> tmp;
            for(int f=0;f<number_nodes[i-1];f++){
                tmp.push_back(powf(-1,f)*0.02*(f+1)*i*(j+1));
            }
            density.push_back(tmp);
        }
        bias.push_back(bias_dens);
        weight_matr.push_back(density);
    }
    return weight_matr;
}

float activateFun(float value){
    return 1/(1+exp(-value));
}
float get_err(vector<float> &predict,vector<float> &trueVal ){
    float err=0;
    for(int i=0;i<predict.size();i++){
        err+=powf(predict[i]-trueVal[i],2);
    }
    cout<<endl<<"error: "<<err/predict.size()<<endl;
}
void changeWeights(vector<vector <vector<float>>>& weight_matr,vector <vector <float>> & bias,vector<float> & in_attr,vector<float>& real_result,float koef_train){
    vector <vector<float>> data;
    data.push_back(in_attr);
    for( int j=0;j<weight_matr.size();j++){
        vector <float> den_res;
        for(int k=0;k<weight_matr[j].size();k++){
            float tmp=0;
            //считаем значение на узле
            for( int m=0;m<data[j].size();m++){
                tmp+=data[j][m]*weight_matr[j][k][m];
            }
            //применяем функцию активации + отступ
            den_res.push_back(activateFun(tmp+bias[j][k]));
        }
        //загоняем значение узлов слоя в вектор
        data.push_back(den_res);
    }
    vector <vector <float>> delta;
    for(int i=data.size()-1;i>0;i--){
        vector <float> tmp;
        for( int j=0;j<data[i].size();j++){
            //производная сигмоиды s'=(1-s)*s
            float delt=data[i][j]*(1-data[i][j]);
            if(delta.size()==0){
                delt*=(-1)*(real_result[j]-data[i][j]);    
            }else{
                //если выходной нейрон
                float delt_sum=0;
                for(int k=0;k<delta[delta.size()-1].size();k++){
                //суммирование дельт, помноженые на весовые коэффиценты слоя, следующего за изменяемым  
                    delt_sum+=delta[delta.size()-1][k]*weight_matr[i][k][j];
                }
                delt*=delt_sum;
            }
            tmp.push_back(delt);
            for(int k=0;k<weight_matr[i-1][j].size();k++){
                //изменение весовых коэффицентов согласно методу обратного распространения ошибок
                weight_matr[i-1][j][k]-=koef_train*delt*data[i-1][j];
            }
            bias[i-1][j]-=koef_train*delt;
        }
        //используем на следующей итерации
        delta.push_back(tmp);
    }
    /*for(auto it:data){
        cout<<"density: "<<endl;
        for(auto it1:it){
            cout << it1<<" ";
        }
        cout<<endl<<endl;
    }*/
    // подсчет ошибки для входного сигнала
        get_err(data[data.size()-1],real_result);
}

float get_acc(vector <vector <float>> &predict,vector <vector<float>> &res){
    int size=res.size();
    int true_val=0;
    float epsilon=0.05;
    for (int i=0;i<size;i++){
        if(fabs((predict[i][0]+predict[i][1])/2-(res[i][0]+res[i][1])/2)<epsilon)
            true_val++;
    }
    float acc=(float)true_val/(float)size;
    cout<<"accuracy: "<<acc; 
    return acc;
}


vector <float> getPredVal(vector<float> & in,vector <vector<vector <float>>> &weights ,vector<vector <float>>& bias){
    vector <float> data=in;
    for(int i=0;i<weights.size();i++){
        vector<float> dens_val;
        for(int j=0;j<weights[i].size();j++){
            float tmp=0;
            for(int k=0;k<weights[i][j].size();k++){
                tmp+=data[k]*weights[i][j][k];
            }
            dens_val.push_back(activateFun(tmp+bias[i][j]));
        }
        data=dens_val;
    }
    return data;
}


float trainModel(int epochs,vector<vector<float>> &attr,vector<vector<float>> & res ,vector<vector<vector<float>>> & weights,vector<vector<float>> &bias,float koef_train){
    float acc;
    for(int i=0;i<epochs;i++){
        cout<<i+1<<" epoch"<<endl;
        vector <vector<float>> predict;
        for(int j=0;j<attr.size();j++){
            //изменение весовых коэффицентов для входного сигнала attr[j]
            changeWeights(weights,bias,attr[j],res[j],koef_train);
        }
        cout<<endl<<"rate: "<<koef_train<<endl;
        //подсчет значений и accuracy на тестовой выборке
        for(int j=0;j<attr.size();j++)
            predict.push_back(getPredVal(attr[j],weights,bias)); 
        acc=get_acc(predict,res); 
        cout<<endl<<endl<<endl;
        //если accuracy>0.6 нас устраивает
        if(acc>0.6)
            return acc;
    }
    return acc;
}

void find_optimal_rate(float rate,float step,vector <vector <float>> & attr,vector <vector <float>> &res,vector <vector <vector<float>>> weights,vector<vector <float>>bias){
    float acc=0;
    vector <vector<vector <float>>> w=weights;
    vector<vector<float>> b=bias;
    while(acc<0.6 && rate<0.5){
        rate+=step;
        weights=w;
        bias=b;
        acc=trainModel(10000,attr,res,weights,bias,rate);
    }
    cout<<rate<<" "<<acc;
}

vector<vector <string>> getData(){
    ifstream file;
    file.open("src/tab.csv");
    string line;
    vector <vector<string>> set;
    while( getline(file,line)){
        stringstream str(line);
        string tmp;
        vector<string>tmp_vec;
        while(getline(str,tmp,','))
            tmp_vec.push_back(tmp);
        set.push_back(tmp_vec);
    }
    return set;
}


// преобразование в реальное значение класса из интервала [0,1]
float getTrueVal(float val,vector <float> min_max){
    return val*(min_max[1]-min_max[0])+min_max[0];
}
vector <float> getResult(vector<float>& input,vector<vector <vector<float>>> & weights,vector<vector <float>> &bias,vector<vector <float>> & min_max,vector<float>&res){
    vector <float> data=getPredVal(input,weights,bias);
    cout<<endl<<"predict result:"<<endl;
    for(int i=0;i<data.size();i++)
        cout<<data[i]<<endl;
    cout<<"real result:"<<endl<<res[0]<<endl<<res[1];
    get_err(data,res);
    cout<<endl;
    return data;
}

vector<vector<float>> getMinMax(vector<vector <float>>& data){
    vector <vector <float>> min_max;
    for(int i=0;i<data[0].size();i++){
        vector <float> min_max_param;
        for(int j=0;j<data.size();j++){
            if(j==0){
                min_max_param.push_back(data[j][i]);
                min_max_param.push_back(data[j][i]);
            }else{
                if(min_max_param[0]>data[j][i]){
                    min_max_param[0]=data[j][i];
                }
                if(min_max_param[1]<data[j][i]){
                    min_max_param[1]=data[j][i];
                }
            }
        }
        //cout<<"min_max "<<min_max_param[0]<<" "<<min_max_param[1]<<endl;
        min_max.push_back(min_max_param);
    }
    return min_max;
}

//нормализуем от 0 до 1
void normalizeData(vector <vector <float>>& data,vector <vector <float>> min_max){
    for (int i=0;i<data.size();i++){
        for(int j=0;j<data[i].size();j++){
            float tmp=(data[i][j]-min_max[j][0])/(min_max[j][1]-min_max[j][0]);
            data[i][j]=tmp;
        }
    }
}

int main(){
    //cout<<"enter attributes"<<endl;
    //сопоставление названий признаков и классов с номерами колонок в векторе с данными
    map<string,int> attr,classes;
    //значения признаков и классов
    vector<vector<float>> values,res;
    string line;
    //getline(cin,line);
    //преобразование данных в удобный вид
    vector <vector<string>> data=getData();
    {
        stringstream ss("Pлин,Pсб1,Ro_c,Дебит воды,Дебит газа,Рзаб1,Рлин1,Рпл. Тек (Карноухов),Рпл. Тек (Расчет по КВД),Рпл. Тек (послед точка на КВД),Руст,Тзаб,Тна шлейфе,Туст,Удельная плотность газа");
        string tmp;
        while(getline(ss,tmp,',')){
            for(int i=0;i<data[1].size();i++){
                if(data[1][i]==tmp){
                    attr[tmp]=i;
                }    
            }
        }
    }
    //cout<<"enter classes"<<endl;
    //getline(cin,line);
    {
        string tmp;
        stringstream ss("G_total,КГФ");
        while(getline(ss,tmp,',')){
            for(int i=0;i<data[1].size();i++){
                if(data[1][i]==tmp){
                    classes[tmp]=i;
                }    
            }
        }
    }
    for(auto row :data){
        vector <float> tmp,tmp1;
        for(auto it : attr){
            stringstream ss(row[it.second]);
            float value;
            if(ss>>value){
                tmp.push_back(value);
            }else{
                tmp.clear();
                break;
            }
        }
        for(auto it1:classes){
            if(row.size()<=it1.second)
                break;
            stringstream ss(row[it1.second]);
            float value;
            if(ss>>value){
                tmp1.push_back(value);
            }else{
                tmp1.clear();
                break;
            }
        }
        if(tmp.size()>0 && tmp1.size()>0){
            values.push_back(tmp); 
            res.push_back(tmp1);
        }
    }
    
    vector <vector <float>> min_max_attr=getMinMax(values);
    vector <vector <float>> min_max_class=getMinMax(res);
    normalizeData(values,min_max_attr);
    normalizeData(res,min_max_class);
    vector <int> number_neur;
    number_neur.push_back(values[0].size());
    number_neur.push_back(values[0].size());
    number_neur.push_back(2);
    vector<vector<float>> bias;
    //инициализация матрицы весов и отступов
    vector <vector <vector<float>>> weights=createModel(bias,values[0].size(),number_neur,0);
    //начальные значения матрицы весов
    cout<<endl<<"weight_matrix begin: "<<endl; 
    for(auto it : weights){
        cout<<"density:"<<endl;
        for(auto it1:it){
            cout<<"node: ";
            for(auto it2:it1){
                cout<< it2<<" ";
            }
            cout<<endl;
        }
        cout<<endl<<endl;
    }
    //обучение
    trainModel(10000,values,res,weights,bias,0.079);
    //нахождение оптимального коэфицента обучения, получили 0.079
    //find_optimal_rate(0.001,0.001,values,res,weights,bias);
     //конечные значения матрицы весов
    cout<<endl<<"weight_matrix end: "<<endl; 
    for(auto it : weights){
        cout<<"density:"<<endl;
        for(auto it1:it){
            cout<<"node: ";
            for(auto it2:it1){
                cout<< it2<<" ";
            }
            cout<<endl;
        }
        cout<<endl<<endl;
    }   
    // проверка работы перцептрона
    vector <vector <float>> predict_values;
    for(int i=0;i<values.size();i++){
        vector <float> pred_res=getResult(values[i],weights,bias,min_max_class,res[i]);
        predict_values.push_back(pred_res);
    }
    get_acc(predict_values,res);
    /*vector<float> input_attr;
    for(auto it:attr){
        while (true){
            cout<<"enter "<<it.first<<endl;
            string tmp;
            cin>>tmp;
            stringstream ss(tmp);
            float val;
            if(ss>>val){
                input_attr.push_back(1/val);
                break;
            }else{
                cout<<"incorrect value";
            }
        }
    }*/

}
