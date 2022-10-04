/**
# ##############################################################################
#  Copyright (c) 2021- University of Bonn                                      #
#  All rights reserved.                                                        #
#                                                                              #
#  Author: Nicky Zimmerman                                                     #
#                                                                              #
#  File: Particle.cpp                			                               #
# ##############################################################################
**/


#include "Particle.h"

Particle::Particle(Eigen::Vector3f p, double w)
{
	pose = p;
	weight = w;
}