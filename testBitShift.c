#include <stdio.h>
#include <stdlib.h>

int main()
{
  unsigned int v;

  while (1)
  {
    printf("Enter number\n");
    scanf("%d",&v);
    printf("V = %d\n",v);

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    printf("Next power of 2 = %d\n",v);
  }
}
