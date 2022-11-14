//Program for sorting elements of any array using fork ()
// gcc -o filename filename.c
//  ./filename
//ps -a1

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>

int cmp (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}


int main()
{

    pid_t pid;
    int status;
    int i;
    int a[] = {6,2,7,24,9};
   
    pid=fork();
    
    if(pid==-1)
    {
        printf("\nError");
    }
    else if(pid>0)
    {
        //sleep(2);
        wait(&status);
        printf("\nParent Process");
        printf("\nParent Process Id:%d",getpid());
        qsort(a,5,sizeof(int),cmp);
    }
    else
    {
        //sleep(5);
        printf("\nChild Process");
        printf("\nChild Process Id: %d ,Parent Process Id:%d.\n",getpid(),getppid());

        qsort(a,5,sizeof(a[0]),cmp);
        printf("\nSorted array:");
        for(i=0; i<5; i++){
            printf("%d\t",a[i]);
        }
        printf("\n\n");
    }
    return 0; 
}