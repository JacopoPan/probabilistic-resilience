#! /usr/local/bin/octave -qf
#Ubuntu's interpreter /usr/bin/octave
#OSX's interpreter /usr/local/bin/octave

function s = vector2string(a)						#convert vectors to strings
	s = "";
	if rows(a) != 1
		printf("ERROR: unacceptable string\n");
		exit;
	elseif length(a) == 0
		printf("WARNING: empty string\n");
	else
		for i=1:(length(a)-1)
			s = strcat(s, int2str(a(1,i)-1));
			s = strcat(s, ", ");
		endfor
		s = strcat(s, int2str(a(1,length(a))-1));
	endif
endfunction

function s_c_j_p_d = sort_cjpd(c_j_p_d)				#sort joint probability distributions by probability values (high to low)
	sorted = [0];
	for i = 1:length(c_j_p_d)
		maximum = -1;
		for j = 1:length(c_j_p_d)
			bool = ismember(sorted, j);
			if (c_j_p_d(j).probability >= maximum) && (!bool)
				maximum = c_j_p_d(j).probability;
				addendum = j;
			endif
		endfor
		sorted = union(sorted, addendum);
		s_c_j_p_d(i) = c_j_p_d(addendum);
	endfor
endfunction



printf("\n\nparameters\n");
N = 20
packet_drop = 0.1
initial_belief = ones(1,N)./N
transition_model = zeros(N,N);
sensor_model = zeros(N,N);

for i=1:N
	for j=1:i
		S = i-1;
		O = j-1;
		sensor_model(i,j) = bincoeff(S,S-O) * packet_drop^(S-O) * (1-packet_drop)^(O);
		#n = S;
		#k = S-O;
		#sensor_model(i,j) = bincoeff(n,k) * packet_drop^(k) * (1-packet_drop)^(n-k);
	endfor
endfor
sensor_model

if(1) #to empirically re-estimate the transition model
for simulations=1:30
	empirical_transition_model = ones(N,N);
	
	#initial random placement
	current_playground = zeros(20,20);
	current_robot_positions = zeros(N,2);
	for robot=1:N
		placed = 0;
		do
			tentative_position  = randi(20, 1, 2);
			if (current_playground(tentative_position(1),tentative_position(2)) == 0)
				current_playground(tentative_position(1),tentative_position(2)) = robot;
				current_robot_positions(robot,:) = tentative_position;
				placed = 1;
			endif
		until (placed == 1)
	endfor
	#count neighbors
	current_neighbors = zeros(N,1);
	for robot=1:N
		for neighbor=1:N
			if (robot != neighbor)
				dist = norm(current_robot_positions(robot,:) - current_robot_positions(neighbor,:), 2);
				if (dist <= 5)
					current_neighbors(robot)++;
				endif
			endif
		endfor
	endfor
	
	for timesteps=1:600
		last_playground = current_playground;
		last_robot_positions = current_robot_positions;
		last_neighbors = current_neighbors;
		
		#move each robot
		for robot=1:N
			tentative_position = current_robot_positions(robot,:);
			random_step = randi(4);
			switch (random_step)
				case 1 tentative_position(1)++;
				case 2 tentative_position(1)--;
				case 3 tentative_position(2)++;
				case 4 tentative_position(2)--;
			endswitch
			if (tentative_position(1) <= 20 && tentative_position(1) >= 1 && tentative_position(2) <= 20 && tentative_position(2) >= 1)
				if (current_playground(tentative_position(1),tentative_position(2)) == 0)
					current_playground(current_robot_positions(robot,1),current_robot_positions(robot,2)) = 0;
					current_playground(tentative_position(1),tentative_position(2)) = robot;
					current_robot_positions(robot,:) = tentative_position;
				endif
			endif
		endfor
		current_playground;
		current_robot_positions;

		#new number of connections of each robot to compare with the previous step
		current_neighbors = zeros(N,1);
		for robot=1:N
			for neighbor=1:N
				if (robot != neighbor)
					dist = norm(current_robot_positions(robot,:) - current_robot_positions(neighbor,:), 2);
					if (dist <= 5)
						current_neighbors(robot)++;
					endif
				endif
			endfor
		endfor
		last_neighbors - current_neighbors;
		
		#add 1s in transition model
		for robot=1:N
			empirical_transition_model(last_neighbors(robot)+1,current_neighbors(robot)+1)++;
		endfor
		
	endfor

	#normalize the transition model by rows
	for row=1:N
		empirical_transition_model(row,:) = empirical_transition_model(row,:)./sum(empirical_transition_model(row,:));
	endfor
	empirical_transition_model;
	transition_model = transition_model + empirical_transition_model;
endfor
#aggregate multiple simulations in one transition model
transition_model = transition_model./30
endif


#experiments

#property 1: "resistance", having always maintained at least 2 connections
printf("\n\nproperty 1: resistance, having always maintained >=10 connections\n");
observations = [10,8,10] 
encoded_observations = observations.+1;
time_horizon = length(observations);

#generate relevant trajectories
printf("\n");
tic();
property_probability = 0;
encoded_state_trajectory = [11,11,11];
for i=1:(10^time_horizon)
	state_trajectory = encoded_state_trajectory.-1;
	
	#linear algortithm
	p_obs_given_seq = 1.0;
	p_seq = 0.0;
	for j=1:time_horizon
		if j == 1
			predicted_belief = initial_belief*transition_model;
			#
			p_obs_given_seq *= sensor_model(encoded_state_trajectory(1),encoded_observations(1));
			p_seq = predicted_belief(encoded_state_trajectory(1));
			predicted_forward = initial_belief*transition_model;
			filtered_forward = predicted_forward.*sensor_model(:,encoded_observations(1))';
		else
			p_obs_given_seq *= sensor_model(encoded_state_trajectory(j),encoded_observations(j));
			p_seq *= transition_model(encoded_state_trajectory(j-1),encoded_state_trajectory(j));
			predicted_forward = filtered_forward*transition_model;
			filtered_forward = predicted_forward.*sensor_model(:,encoded_observations(j))';
		endif
	endfor
	p_obs = sum(filtered_forward);
	property_probability += (p_obs_given_seq*p_seq)/p_obs;
	
	encoded_state_trajectory(1,time_horizon)++;												#next state trajectory
	for k=1:(time_horizon-1)
		if encoded_state_trajectory(1,time_horizon-k+1) > 20
			encoded_state_trajectory(1,time_horizon-k+1) = 11;
			encoded_state_trajectory(time_horizon-k)++;
		endif
	endfor
endfor
property_probability
TIME_algo_p1 = toc()


#property 2: having lost connectivity during the last move
printf("\n\nproperty 2: having lost connectivity during the last move\n");
observations = [2,1,0] #plausible observations: [2,2,1,0], [3,2,1,0], [2,2,2,0], [1,1,1,0]
encoded_observations = observations.+1;
time_horizon = length(observations);

#generate relevant trajectories
printf("\n");
tic();
property_probability = 0;
updating_state_trajectory = [2,2];
for i=1:((20-1)^(time_horizon-1))
	updating_state_trajectory;
	encoded_state_trajectory = horzcat(updating_state_trajectory,[1]);
	state_trajectory = encoded_state_trajectory.-1;
	
	#linear algortithm
	p_obs_given_seq = 1.0;
	p_seq = 0.0;
	for j=1:time_horizon
		if j == 1
			predicted_belief = initial_belief*transition_model;
			#
			p_obs_given_seq *= sensor_model(encoded_state_trajectory(1),encoded_observations(1));
			p_seq = predicted_belief(encoded_state_trajectory(1));
			predicted_forward = initial_belief*transition_model;
			filtered_forward = predicted_forward.*sensor_model(:,encoded_observations(1))';
		else
			p_obs_given_seq *= sensor_model(encoded_state_trajectory(j),encoded_observations(j));
			p_seq *= transition_model(encoded_state_trajectory(j-1),encoded_state_trajectory(j));
			predicted_forward = filtered_forward*transition_model;
			filtered_forward = predicted_forward.*sensor_model(:,encoded_observations(j))';
		endif
	endfor
	p_obs = sum(filtered_forward);
	property_probability += (p_obs_given_seq*p_seq)/p_obs;
	
	updating_state_trajectory(1,time_horizon-1)++;												#next state trajectory
	for k=1:(time_horizon-2)
		if updating_state_trajectory(1,time_horizon-k) > 20
			updating_state_trajectory(1,time_horizon-k) = 2;
			updating_state_trajectory(time_horizon-k-1)++;
		endif
	endfor
endfor
property_probability
TIME_algo_p2 = toc()