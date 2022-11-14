#include <stdio.h>
#include <stdlib.h>
int main(){
    
    int arrival_time [] ={1,2,3,4};
    int burst_time [] ={4,4,5,8};
    int temp[4];
    int i,smallest,count=0,limit = 4,time;
    double wait_time=0,turnaround_time=0,end;
    
    for(i=0;i<limit;i++){
        temp[i]=burst_time[i];
    }
    burst_time[9]=9999;
    
    for(time=0;count!=limit;time++){
        smallest=9;
        for(i=0;i<limit;i++){
            if(arrival_time[i]<=time && burst_time[i]<burst_time[smallest] && burst_time[i]>0){
                smallest=i;
            }
            
        }
        burst_time[smallest]--;
        if(burst_time[smallest]==0){
                count++;
                end=time+1;
                wait_time=wait_time+end-arrival_time[smallest]-temp[smallest];
                turnaround_time=turnaround_time+end-arrival_time[smallest];
                
                
        }
    }
    float average_waiting_time=wait_time/limit;
    float average_turnaround_time=turnaround_time/limit;
    printf("\n Average Waiting Time is %lf",average_waiting_time);
    printf("\n Average turnaround time is %lf",average_turnaround_time);
    return 0;
}