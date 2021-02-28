#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "exception.cpp"
#include <time.h>
#include <vector>
#include "PHARMclass.h"
#include "submodel.h"
#include "config.cpp"

typedef unsigned int uint;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

void printInfo(){
    std::cout << "Extracts PHARM features from all greyscale JPEG images in the directory input-dir.\n";
	std::cout << "Author: Vojtech Holub.\n";
	std::cout << "For further details read:\n";
	std::cout << "   Phase-Aware Projection Model for Steganalysis in JPEG Images, V. Holub and J. Fridrich, SPIE 2015\n\n";
	std::cout << "usage: PHARM -Q -I input-dir -O output-dir [-S] [-n] [-T] [-s] [-q] \n\n";
}

void WriteFeaToFiles(PHARMclass *PHARMobj, std::string oDir, bool verbose)
{
	if (verbose) std::cout << std::endl << "---------------------" << std::endl << "Writing features to the output directory" << std::endl;
	std::vector<Submodel *> submodels = PHARMobj->GetSubmodels();
	for (int submodelIndex=0; submodelIndex < (int)submodels.size(); submodelIndex++)
	{
		Submodel *currentSubmodel = submodels[submodelIndex];
		std::string submodelName = currentSubmodel->modelName;
		if (verbose) std::cout << "   " << submodelName << std::endl;

		fs::path dir (oDir);
		fs::path file (submodelName);
		std::string full_path = ((dir / file).string()+ ".fea");

		if (fs::exists(full_path)) fs::remove(full_path);

		std::ofstream outputFile;
		outputFile.open(full_path.c_str());
		for (int imageIndex=0; imageIndex < (int)currentSubmodel->ReturnFea().size(); imageIndex++)
		{
			float *currentLine = (currentSubmodel->ReturnFea())[imageIndex];
			for (int feaIndex=0; feaIndex < currentSubmodel->dim; feaIndex++)
			{
				outputFile << currentLine[feaIndex] << " ";
			}
			outputFile << PHARMobj->imageNames[imageIndex] << std::endl;
		}
		outputFile.close();
	}
}

int main(int argc, char** argv)
{
	try { 
		std::string iDir, oDir;
		std::vector< std::string > images;
		bool verbose = false;
		int QF = -1;
		int nu, T, s, seed;
		float q;
		po::variables_map vm;

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
			("verbose,v",       po::bool_switch(&verbose),                     "print out verbose messages")
            ("input-dir,I",     po::value<std::string>(&iDir),                 "directory with images to calculate the features from")
            ("images,i",        po::value<std::vector<std::string> >(&images), "list of images to calculate the features from")
			("output-dir,O",    po::value<std::string>(&oDir),                 "dir to output features from all submodels")
			("quality factor,Q",po::value<int>(&QF),                           "image quality factor (standard quantization matrix)")
			("seed,S",			po::value<int>(&seed)->default_value(1),       "seed for generating random projections")
			("n",				po::value<int>(&nu)->default_value(900),         "number of projections per residual")
			("T",				po::value<int>(&T)->default_value(2),          "number of bins per projection")
			("s",				po::value<int>(&s)->default_value(8),          "maximum size of the projection neighbor-hood")
			("q",				po::value<float>(&q)->default_value(-1),       "override quantization based on quality factor and use this one")
        ;
        po::positional_options_description p;
        p.add("cover-images", -1);

        po::store(po::command_line_parser(argc,argv).options(desc).positional(p).run(), vm);
        po::notify(vm);

        if (vm.count("help"))  { printInfo(); std::cout << desc << "\n"; return 1; }
        if (!vm.count("output-dir")){ std::cout << "'output-dir' is required." << std::endl << std::endl << desc << std::endl; return 1; }
		else if (!fs::is_directory(fs::path(oDir))) { std::cout << "'output-dir' must be an existing directory." << std::endl << std::endl << desc << std::endl; return 1; }
		if (QF == -1) { std::cout << "'Q' is required." << std::endl << std::endl << desc << std::endl; return 1; }
		if (QF < 1) { std::cout << "'Q' must be greater greater than zero." << std::endl << std::endl << desc << std::endl; return 1; }
		if (QF > 100) { std::cout << "'Q' must be smaller or equal than 100." << std::endl << std::endl << desc << std::endl; return 1; }
		if (nu < 1) { std::cout << "'n' must be greater than zero." << std::endl << std::endl << desc << std::endl; return 1; }
		if (T < 1) { std::cout << "'T' must be greater than zero." << std::endl << std::endl << desc << std::endl; return 1; }
		if (s < 1) { std::cout << "'s' must be greater than zero." << std::endl << std::endl << desc << std::endl; return 1; }
		if ((q <= 0) && (q!=-1)) { std::cout << "'q' must be greater than zero." << std::endl << std::endl << desc << std::endl; return 1; }

		// add all jpeg files from the input directory to the vector
		fs::directory_iterator end_itr; // default construction yields past-the-end
		if (vm.count("input-dir"))
			for ( fs::directory_iterator itr(iDir); itr!=end_itr; ++itr ) 
			{
				if ( (!fs::is_directory(itr->status())) && (itr->path().extension()==".jpg") )
					images.push_back(itr->path().string());
            }

		// create config object
		Config *config = new Config(verbose, QF, seed, nu, T, s, q);
		
		// create object with all the submodels and compute the features
		PHARMclass *PHARMobj = new PHARMclass(config);

		// Run the feature computation
		PHARMobj->ComputeFeatures(images);

		// writes features from all the submodels to the separate files
		WriteFeaToFiles(PHARMobj, oDir, verbose);

		// Remove PHARMobj from the memory
		delete PHARMobj;
		delete config;
    } 
	catch(std::exception& e) 
	{ 
		std::cerr << "error: " << e.what() << "\n"; return 1; 
	} 
	catch(...) 
	{ 
		std::cerr << "Exception of unknown type!\n"; 
	}		
}
