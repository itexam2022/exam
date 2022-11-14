#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

int cmp (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

int main()
{
    pid_t pid;
    int i,n;
    int val[] = {2,6,3,7,9,5};

    qsort(val,6,sizeof(int),cmp);
    
    printf("\nSorted elements are: ");
    for(i=0;i<n;i++){
        printf("\t%d",val[i]);
    }
    char *args[]={"./2B2", val, NULL};
    pid=fork();
    if(pid==0)
    {
        execv(args[0], args);
        perror("Error in execve call..."); 
    }
    return val;
}
