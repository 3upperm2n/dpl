#ifndef ARGOPT_H
#define ARGOPT_H

#include <stdio.h> 
#include <stdlib.h>
#include <getopt.h>

extern char *optarg;
extern int optind, opterr, optopt;

static struct option full_options[] = {
  { "width",  required_argument, 0, 'w'},
  { "height", required_argument, 0, 'h'},
  { 0, 0, 0, 0 }                                                          
};  

static const char *arg_help =
"\n"
"\tfwd_conv [<options>] <sources>\n"                                             
"\n"                                                                        
"Options:\n"                                                                
"\n"                                                                        
"-w\n"                                                                   
"\timage width.\n"                                      
"-h\n"                                                                   
"\timage height.\n"                                      
"\n";

class ARG 
{
public:
  int opt;                                                                    
  int option_index;

  int width;
  int height;

  ARG() { option_index = 0;}

  void read_cmd_line(int argc, char** argv)
  {
	if (argc == 1)                                                              
	{                                                                           
	  printf("%s", arg_help);
	  exit(0);                                                                
	} 		

	while ((opt = getopt_long(argc, argv, "w:h:", full_options, &option_index)) != -1)
	{                                                                           
	  switch(opt)
	  {
		case 0:
		  if (full_options[option_index].flag != 0)
			break;
		  printf ("option %s", full_options[option_index].name);
		  if (optarg)
			printf (" with arg %s", optarg);
		  printf ("\n");
		  break;

		case 'w':
		  width = atoi(optarg);
		  break;

		case 'h':
		  height = atoi(optarg);
		  break;

		case '?':
		  break;

		default:
		  abort ();
	  }
	}
  }
};
#endif
