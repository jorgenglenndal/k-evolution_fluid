#include <iomanip>
#include <iostream>

int main(){

    int N_old = 10;
    int N_new = 11;
    int I = 3;
    int test = N_new*(1. - (I+0.)/N_old);
    //int test2 = N_new*(i+1.)/N_old;
    if (N_new % N_old == 0) std::cout << "no remainder" << std::endl;
   // std::cout << test << std::endl;

    return 0;
}