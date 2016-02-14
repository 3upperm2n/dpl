#ifndef ARGOPT_H
#define ARGOPT_H

#include <stdio.h> 
#include <stdlib.h>

const char short_opt[] = {'w', 'h', 'u', '\0'};
const char *long_opt[] = {"--width", "--height", "--usage"};

void usage(char *argv)
{
  printf("Usage:\n");
  printf("\t%s opt opt_val\n\n", argv);
  printf(" -w    (--width)         image_width\n");
  printf(" -h    (--height)        image_height\n");
}

class ARG 
{
public:
  int opt;                                                                    
  int option_index;

  struct IMAGE {
	int width; 
	int height;
  } image;

  ARG() { 
	option_index = 0;
	image.width  = -1; 
	image.height = -1; 
  }

  int moreopt(char *argv);
  int read_opt(int argc, char **argv, int id, void *data, const char *datatype);
  void parsing(int argc, char **argv);
};

int ARG::moreopt(char *argv) {
  int i=0;
  while(short_opt[i]!='\0'){
	if(strcmp(argv, long_opt[i])==0){
	  argv[1]=short_opt[i];
	  argv[2]='\0';
	  return 0;
	}
	i++;
  }
  return 1;
}

int ARG::read_opt(int argc, char **argv, int id, void *data, const char *datatype)
{
  if(strcmp(datatype, "int")==0) {
	if(id+1 >= argc) {
	  fprintf(stderr, "incomplete input for %s\n", argv[id]);
	  usage(argv[0]);
	  exit(EXIT_FAILURE);
	}
	*((int*)data)=atoi(argv[id+1]);
  }
  return id+1;
}

void ARG::parsing(int argc, char **argv)
{
  if(argc<=1){
	usage(argv[0]);
	exit(EXIT_FAILURE);
  }

  int i=1;
  while(i < argc)
  {
	if(argv[i][0]=='-') 
	{
	  if(argv[i][1]=='-'){
		// read long options
		if(moreopt(argv[i]))
		  fprintf(stderr,"unknown verbose option : %s\n", argv[i]);
	  }	

	  // read short options
	  switch(argv[i][1])
	  {
		case 'u':
		  usage(argv[0]);
		  exit(EXIT_FAILURE);
	  
		case 'w':
		   i=read_opt(argc, argv, i, &image.width, "int");
		  break;

		case 'h':
		  i=read_opt(argc, argv, i, &image.height, "int");
		  break;
	  }
	}
	i++;
  }
}


#endif
