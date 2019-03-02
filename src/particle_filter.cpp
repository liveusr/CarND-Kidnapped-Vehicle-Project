/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::default_random_engine;
using std::normal_distribution;
using std::discrete_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  num_particles = 50;  // TODO: Set the number of particles

  // initialize the random engine
  default_random_engine gen;

  // normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // initialize all particles
  for(int i = 0; i < num_particles; i++) {
    Particle p;

    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
    weights.push_back(p.weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // initialize the random engine
  default_random_engine gen;

  // normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  // calculate new position and heading of all particles
  for(int i = 0; i < num_particles; i++) {
    double new_x, new_y, new_theta;

    if(fabs(yaw_rate) > 0.0001) {
      // if yaw_rate is not equal to zero
      new_x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      new_y = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      new_theta = particles[i].theta + yaw_rate * delta_t;
    }
    else {
      // if yaw rate is equal to zero
      new_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      new_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
      new_theta = particles[i].theta;
    }

    // add Gaussian noise and update new predictions
    particles[i].x = new_x + dist_x(gen);
    particles[i].y = new_y + dist_y(gen);
    particles[i].theta = new_theta + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for(unsigned int i = 0; i < observations.size(); i++) {
    double dist = -1;

    for(unsigned int j = 0; j < predicted.size(); j++) {
      double dx = predicted[j].x - observations[i].x;
      double dy = predicted[j].y - observations[i].y;
    
      // find the nearest neighbor
      if((dist == -1) || ((dx*dx + dy*dy) < dist)) {
        dist = dx*dx + dy*dy;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // calculate normalization term
  double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  // calculate divisor required for calculating exponent
  double exp_divisor_x = 2 * std_landmark[0] * std_landmark[0];
  double exp_divisor_y = 2 * std_landmark[1] * std_landmark[1];

  // for every particle in range
  for(int i = 0; i < num_particles; i++) {
    Particle p = particles[i];

    // find landmarks in range
    vector<LandmarkObs> lm_in_range;
    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      LandmarkObs lm = LandmarkObs{map_landmarks.landmark_list[j].id_i,
                                   map_landmarks.landmark_list[j].x_f,
                                   map_landmarks.landmark_list[j].y_f};
      double dx = p.x - lm.x;
      double dy = p.y - lm.y;

      if((dx*dx + dy*dy) <= (sensor_range*sensor_range)) {
        lm_in_range.push_back(lm);
      }
    }

    // transform observations from car coordinates to map coordinates
    vector<LandmarkObs> obs_mapped;
    for(unsigned int j = 0; j < observations.size(); j++) {
      double xm = p.x + (cos(p.theta) * observations[j].x) - (sin(p.theta) * observations[j].y);
      double ym = p.y + (sin(p.theta) * observations[j].x) + (cos(p.theta) * observations[j].y);
      obs_mapped.push_back(LandmarkObs{observations[j].id, xm, ym});
    }

    // associate each transformed observation with a landmark identifier
    dataAssociation(lm_in_range, obs_mapped);
    
    // calculate particle's final weight
    double weight = 1.0;
    for(unsigned int j = 0; j < obs_mapped.size(); j++) {
      for(unsigned int k = 0; k < lm_in_range.size(); k++) {
        if(lm_in_range[k].id == obs_mapped[j].id) {
          double dx = lm_in_range[k].x - obs_mapped[j].x;
          double dy = lm_in_range[k].y - obs_mapped[j].y;
          weight *= gauss_norm * exp( -(((dx*dx) / exp_divisor_x) + ((dy*dy) / exp_divisor_y)) );
          break;
        }
      }
    }
    particles[i].weight = weight;
    weights[i] = weight;

    // For visualization (blue lines on simulator)
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
    for(unsigned int j = 0; j < obs_mapped.size(); j++) {
      associations.push_back(obs_mapped[j].id);
      sense_x.push_back(obs_mapped[j].x);
      sense_y.push_back(obs_mapped[j].y);
    }
    SetAssociations(particles[i], associations, sense_x, sense_y);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // initialize the random engine
  default_random_engine gen;

  vector<Particle> new_particles;

  // generate distribution proportional to weight
  discrete_distribution<int> dist(weights.begin(), weights.end());

  // resample particles using above distribution
  for(int i = 0; i < num_particles; i++) {
    new_particles.push_back(particles[dist(gen)]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
