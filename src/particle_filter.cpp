/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	num_particles = 40;

	for(unsigned int i=0; i< num_particles; i ++){
		Particle particle;
		particle.id = i;
		particle.x = x;
		particle.y = y;
		particle.theta = theta;
		particle.weight = 1.0;

		particle.x += dist_x(gen);
		particle.y += dist_y(gen);
		particle.theta += dist_theta(gen);

		particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);

	for(unsigned int i = 0; i< num_particles; i++){
		Particle p = particles[i];
		if(yaw_rate != 0){
			p.x = p.x + velocity/yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
			p.y = p.y + velocity/yaw_rate * (cos(p.theta) - cos(p.theta - yaw_rate*delta_t));
			p.theta = p.theta + yaw_rate*delta_t;
		}
		else{
			p.x = p.x + velocity * delta_t * cos(p.theta);
			p.y = p.y + velocity * delta_t * sin(p.theta);
		}

		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); i++) { 
		LandmarkObs o = observations[i];
		double min_dist = numeric_limits<double>::max();
		int predicted_map_id = -1;
		
		for (unsigned int j = 0; j < predicted.size(); j++) {
			LandmarkObs p = predicted[j];
			
			double cur_dist = dist(o.x, o.y, p.x, p.y);
			if (cur_dist < min_dist) {
				min_dist = cur_dist;
				predicted_map_id = p.id;
			}
		}

		o.id = predicted_map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for(unsigned int i=0; i<num_particles; i++){
		Particle p = particles[i];
		vector<LandmarkObs> predictions;

		for(unsigned int j=0; j<map_landmarks.landmark_list.size(); j++){
			Map::single_landmark_s ml = map_landmarks.landmark_list[i];

			double cur_dist = dist(ml.x_f, ml.y_f, p.x, p.y);
			if (cur_dist <= sensor_range) {
				predictions.push_back(LandmarkObs{ ml.id_i, ml.x_f, ml.y_f });
			}
		}

		vector<LandmarkObs> map_observations;
		for (unsigned int j = 0; j < observations.size(); j++) {
			LandmarkObs o = observations[j];
			double tX = cos(p.theta)*o.x - sin(p.theta)*o.y + p.x;
			double tY = sin(p.theta)*o.x + cos(p.theta)*o.y + p.y;
			map_observations.push_back(LandmarkObs{ o.id, tX, tY });
		}

		dataAssociation(predictions, map_observations);

		p.weight = 1.0;

		for (unsigned int j = 0; j < map_observations.size(); j++) {
			LandmarkObs o = map_observations[i];

			LandmarkObs nearest_prediction;
			for (unsigned int k = 0; k < predictions.size(); k++) {
				LandmarkObs p = predictions[k];
				if (p.id == o.id) {
				nearest_prediction.x = p.x;
				nearest_prediction.y = p.y;
				}
			}

			double sX = std_landmark[0];
			double sY = std_landmark[1];
			double obs_w = ( 1/(2*M_PI*sX*sY)) * exp( -( pow(nearest_prediction.x-o.x,2)/(2*pow(sX, 2)) + (pow(nearest_prediction.y-o.y,2)/(2*pow(sY, 2))) ) );

			p.weight *= obs_w;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
  vector<Particle> new_particles;
  vector<double> weights;

  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);
  double max_weight = *max_element(weights.begin(), weights.end());
  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;
  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
