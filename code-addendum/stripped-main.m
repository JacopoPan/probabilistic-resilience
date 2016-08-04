#! /usr/bin/octave -qf
#Ubuntu's interpreter /usr/bin/octave
#OSX's interpreter /usr/local/bin/octave

function [g_b, t_m_a, s_m_a, u_v] = choose_model(s)
g_b = [0.33, 0.34, 0.33];		#P(healthy, sick, very sick)
t_m_a(1).val = [0.8, 0.15, 0.05;	#P(healthy, sick, very sick | was healthy)
				0.2, 0.7, 0.1;		#P(healthy, sick, very sick | was sick)
				0.1, 0.3, 0.6];		#P(healthy, sick, very sick | was very sick)
s_m_a(1).val = [0.9, 0.05, 0.05;	#P(normal, warm, very warm | is healthy)
				0.2, 0.7, 0.1;		#P(normal, warm, very warm | is sick)
				0.1, 0.6, 0.3];		#P(normal, warm, very warm | is very sick)
u_v = [1, 0, -2];			#U(healthy, sick, very sick)
endfunction

function integrity_checks(o_v, g_b, t_m_a, s_m_a, u_v)						#integrity checks on the input parameters
	state_card = columns(t_m_a(1).val);
	observation_card = columns(s_m_a(1).val);
	if (state_card < 1) || (observation_card < 1) 						#on the existance of states and observations
		printf("ERROR: impossible observation vector\n");
		exit;
	endif
	for i=1:length(o_v)									#on the observation vector
		if o_v(i) < 0 || o_v(i) > observation_card
			printf("ERROR: impossible observation vector\n");
			exit;
		endif
	endfor
	if ((state_card^length(o_v)*observation_card^length(o_v)) > 1000000)			#on the size of the joint distribution
		printf("WARNING: over a million entries in the joint distribution\n");
	endif
	if (length(g_b) != state_card) || (abs(sum(g_b)-1) > 0.000001)				#on the ground belief
		printf("ERROR: impossible ground belief\n");
		exit;
	endif
	for i=1:length(t_m_a)									#on the transition model array
		if (columns(t_m_a(i).val) != state_card) || (rows(t_m_a(i).val) != state_card)
			printf("ERROR: impossible %d-th transition model\n", i);
			exit;
		endif
		for j=1:rows(t_m_a(i).val)
			if abs(sum(t_m_a(i).val(j,:))-1) > 0.000001
				printf("ERROR: impossible %d-th transition model, %d-th row\n", i, j);
				exit;
			endif
		endfor
	endfor
	for i=1:length(s_m_a)									#on the sensor model array
		if (columns(s_m_a(i).val) != observation_card) || (rows(s_m_a(i).val) != state_card)
			printf("ERROR: impossible %d-th sensor model\n", i);
			exit;
		endif
		for j=1:rows(s_m_a(i).val)
			if abs(sum(s_m_a(i).val(j,:))-1) > 0.000001
				printf("ERROR: impossible %d-th transition model, %d-th row\n", i, j);
				exit;
			endif
		endfor
	endfor
	if (length(u_v) != state_card)								#on the utility vector
		printf("ERROR: impossible utility vector\n");
		exit;
	endif
endfunction

function bool = compatible_trajectories(o_t, o_v)			#true IFF two trajectories CAN be the same due to missing obs
	bool = 1;
	for i =1:min(length(o_t),length(o_v))
		if (o_t(1,i) != o_v(1,i)) && (o_v(1,i) != 0)		#incompatible IFF different && !because of a missing obs
			bool = 0;
			break;
		endif
	endfor
endfunction

function bool = identical_trajectories(o_t, o_v)			#true IFF two trajectories are EXACTLY the same
	bool = 1;
	if length(o_t) != length(o_v)
		printf("ERROR: observations of different length\n");
		exit;
	endif
	for i =1:max(length(o_t),length(o_v))
		if (o_t(1,i) != o_v(1,i))
			bool = 0;
			break;
		endif
	endfor
endfunction

function [c_j_p_d, t_p] = compute_cjpd(g_b, t_m_a, s_m_a, o_v, u_v)				#compute the CONDITIONAL jpd
	state_card = columns(t_m_a(1).val);
	observation_card = columns(s_m_a(1).val);
	t_h = length(o_v);
	entry = 1;
	state_trajectory = ones(1,t_h);
	total_probability = 0;
	for i=1:state_card^t_h									#for each trajectory
		observation_trajectory = ones(1,t_h);
		c_j_p_d(entry).state_trajectory = state_trajectory;
		c_j_p_d(entry).observation_trajectory = o_v;
		c_j_p_d(entry).utility_trajectory = zeros(1,t_h);				#get utility trajectory
		for j=1:t_h
			c_j_p_d(entry).utility_trajectory(1,j) = u_v(1,state_trajectory(1,j));
		endfor										
		c_j_p_d(entry).probability = 0;
		for j=1:observation_card^t_h							#for each observation trajectory
			if compatible_trajectories(observation_trajectory, o_v)			#update entry IFF compatible
				temporary_probability = (g_b*t_m_a(1).val)(1,state_trajectory(1,1))*s_m_a(1).val(state_trajectory(1,1),observation_trajectory(1,1));
				for k=2:t_h
					temporary_probability *= t_m_a(1).val(state_trajectory(1,k-1),state_trajectory(1,k));
					temporary_probability *= s_m_a(1).val(state_trajectory(1,k),observation_trajectory(1,k));
				endfor
				c_j_p_d(entry).probability += temporary_probability;
				total_probability += temporary_probability;			#total probabilty, for normalization	
			endif

			observation_trajectory(1,t_h)++;					#next observation trajectory
			for k=1:(t_h-1)
				if observation_trajectory(1,t_h-k+1) > observation_card
					observation_trajectory(1,t_h-k+1) = 1;
					observation_trajectory(t_h-k)++;
				endif
			endfor
		endfor
		entry++;
		state_trajectory(1,t_h)++;							#next state trajectory
		for k=1:(t_h-1)
			if state_trajectory(1,t_h-k+1) > state_card
				state_trajectory(1,t_h-k+1) = 1;
				state_trajectory(t_h-k)++;
			endif
		endfor
	endfor
	for i =1:length(c_j_p_d)								#normalize distribution
		c_j_p_d(i).probability = c_j_p_d(i).probability/total_probability;
	endfor
	t_p = total_probability;
endfunction

function s = vector2string(a)					#convert vectors to strings
	s = "";
	if rows(a) != 1
		printf("ERROR: unacceptable string\n");
		exit;
	elseif length(a) == 0
		printf("WARNING: empty string\n");
	else
		for i=1:(length(a)-1)
			s = strcat(s, int2str(a(1,i)));
			s = strcat(s, ", ");
		endfor
		s = strcat(s, int2str(a(1,length(a))));
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

function print_cjpd(c_j_p_d)							#print joint probability distributions
	for i = 1:length(c_j_p_d)
		printf("P(x=[%s]|o=[%s]) = %f\n", vector2string(c_j_p_d(i).state_trajectory(1,:)), vector2string(c_j_p_d(i).observation_trajectory(1,:)), c_j_p_d(i).probability);
	endfor
endfunction

function o_v = observation_vector_disambiguation(o_v_input, s_m_a)			#used when the observations are autogenerated
	if strcmp ("random", o_v_input)
		time_horizon = 4;							#constant for safety reasons
		o_v = randi(columns(s_m_a(1).val), 1, time_horizon);			#random observations
	else
		o_v = o_v_input;
	endif
endfunction

for iter=1:1
	printf("\nExperiment %d\n", iter);

	[ground_belief, transition_model_array, sensor_model_array, utility_vector] = choose_model(model_choice);
	disambiguated_observation_vector = observation_vector_disambiguation(observation_vector, sensor_model_array);

	integrity_checks(disambiguated_observation_vector, ground_belief, transition_model_array, sensor_model_array, utility_vector);

	[conditional_joint_probability_distribution, total_probability] = compute_cjpd(ground_belief, transition_model_array, sensor_model_array, disambiguated_observation_vector, utility_vector);

	sorted_conditional_joint_probability_distribution = sort_cjpd(conditional_joint_probability_distribution);
	printf("Actual CJPD\n");
	print_cjpd(sorted_conditional_joint_probability_distribution);
	
endfor
printf("\n");