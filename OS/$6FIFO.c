#include<stdio.h>
int main()
{
    int i,frame[10],avail,count=0;
    int n = 20;
    int a[] = {7,0,1,2,0,3,0,4,2,3,0,3,2,1,2,0,1,7,0,1};
    int no = 3;
    
    for(i=0;i<no;i++){
        frame[i]= -1;
    }

    int k,j=0;
    printf("\tref string\t\t page frames\n");
    for(i=0;i<n;i++)
    {
        printf("%d\t\t",a[i]);
        avail=0;
        for(k=0;k<no;k++){
            if(frame[k]==a[i]){
                avail=1;
            }
        }

        if (avail==0)
        {
            frame[j]=a[i];
            j=(j+1)%no;
            // j++;
            // if(j == 3) j = 0;
            count++;
            for(k=0;k<no;k++)
                printf("%d\t",frame[k]);
        }
        printf("\n");

    }

    printf("\nPage Fault Is %d",count);
    return 0;

}
