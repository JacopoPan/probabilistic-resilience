#! /usr/bin/octave -qf
#Ubuntu's interpreter /usr/bin/octave
#OSX's interpreter /usr/local/bin/octave

#################################################
#################################################
# PARAMETERS ####################################
#################################################
#################################################

model_choice = "rain";		#"rain" for the rain model, "sick" for the sickness model, "random" for autogeneration
observation_vector = [1, 1];	#integers referring to columns in the sensor model, use "0" for <missing>, or "random" for autogeneration
experiments = 1;		#number of times the "main" is repeated

#################################################
#################################################
# HMMs FUNCTIONS ################################
#################################################
#################################################

function [g_b, t_m_a, s_m_a, u_v] = choose_model(s)	#model choice
	switch s
	case "rain"					#rain model
		g_b = [0.5, 0.5];			#P(rain, no rain)
		t_m_a(1).val = [0.7, 0.3;		#P(rain, no rain | it was raining)
				0.3, 0.7];		#P(rain, no rain | it wasn't raining)
		s_m_a(1).val = [0.9, 0.1;		#P(umbrella, no umbrella | it is raining)
				0.2, 0.8];		#P(umbrella, no umbrella | it isn't raining)
		u_v = [0, 1];				#U(rain, no rain)
	
	case "sick"					#sickness model
		g_b = [0.33, 0.34, 0.33];		#P(healthy, sick, very sick)
		t_m_a(1).val = [0.8, 0.15, 0.05;	#P(healthy, sick, very sick | was healthy)
				0.2, 0.7, 0.1;		#P(healthy, sick, very sick | was sick)
				0.1, 0.3, 0.6];		#P(healthy, sick, very sick | was very sick)
		s_m_a(1).val = [0.9, 0.05, 0.05;	#P(normal, warm, very warm | is healthy)
				0.2, 0.7, 0.1;		#P(normal, warm, very warm | is sick)
				0.1, 0.6, 0.3];		#P(normal, warm, very warm | is very sick)
		u_v = [1, 0, -2];			#U(healthy, sick, very sick)
	
	case "random"					#random model
		state_card = randi(5);			#up to 5 states
		observation_card = randi(5);		#up to 5 observations
		printf("WARNING: random model with %d states and %d observations\n", state_card, observation_card);
		g_b = rand(1,state_card);
		g_b = g_b./sum(g_b);
		temp_t_m = zeros(state_card, state_card);
		temp_s_m = zeros(state_card, observation_card);
		for i = 1:state_card
			temp_t_m(i,:) = rand(1,state_card);
			temp_t_m(i,:) = temp_t_m(i,:)./sum(temp_t_m(i,:));
			temp_s_m(i,:) = rand(1,observation_card);
			temp_s_m(i,:) = temp_s_m(i,:)./sum(temp_s_m(i,:));
		endfor
		t_m_a(1).val = temp_t_m;
		s_m_a(1).val = temp_s_m;
		u_v = randi(10, 1, state_card);		#utility values in the range 1-10		
	
	otherwise
		printf("ERROR: impossible model choice\n");
		exit;
	endswitch
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

function t_a_e = compare_sorted_cjpds(c_j_p_d_1, c_j_p_d_2)		#compute the L1 (Manhattan) distance between two distributions
	t_a_e = 0.0;							#defined over (possibly non identical) domains
	elements_of_one_missing_in_two = 0;
	for i=1:length(c_j_p_d_1)
		found = false;
		for j=1:length(c_j_p_d_2)
			if length(c_j_p_d_1(i).state_trajectory) != length(c_j_p_d_2(j).state_trajectory)
				printf("ERROR: incompatible conditional joint probability distributions\n");
				exit;
			endif
			if !identical_trajectories(c_j_p_d_1(i).observation_trajectory, c_j_p_d_2(j).observation_trajectory)
				printf("ERROR: incompatible conditional joint probability distributions\n");
				exit;
			endif
			if identical_trajectories(c_j_p_d_1(i).state_trajectory, c_j_p_d_2(j).state_trajectory)
				t_a_e += abs(c_j_p_d_1(i).probability-c_j_p_d_2(j).probability);
				found = true;
			endif
		endfor
		if found == false;					#account of elements in JPD1 that are not in JPD2
			elements_of_one_missing_in_two++;
			t_a_e += c_j_p_d_1(i).probability;
		endif
	endfor
	
	elements_of_two_missing_in_one = 0;
	for i=1:length(c_j_p_d_2)
		found = false;
		for j=1:length(c_j_p_d_1)
			if identical_trajectories(c_j_p_d_2(i).state_trajectory, c_j_p_d_1(j).state_trajectory)
				found = true;
			endif
		endfor
		if found == false;					#account of elements in JPD2 that are not in JPD1
			elements_of_two_missing_in_one++;
			t_a_e += c_j_p_d_2(i).probability;
		endif
	endfor
	elements_of_two_missing_in_one;
	elements_of_one_missing_in_two;
endfunction

function [c_j_p_d, c_p] = compute_heuristic_cjpd(g_b, t_m_a, s_m_a, o_v, u_v) 		#Heuristic search of the ML trajectories 
	state_card = columns(t_m_a(1).val);						#BIG DISCLAIMER: the Viterbi algorithm is
	observation_card = columns(s_m_a(1).val);					#NOT meant to work with missing observations
	t_h = length(o_v);
	covered_probability = 0;							#this is the fraction of the CJPD covered
														
	c_j_p_d(1).state_trajectory = ones(1,t_h); 
	c_j_p_d(1).observation_trajectory = o_v; 
	c_j_p_d(1).utility_trajectory = zeros(1,t_h); 
	c_j_p_d(1).probability = 0.0;
	
	#TO BE IMPLEMENTED
	#compute most likely traj using the viterbi algorithm
	#compute its probability using the forward algorithm and the bayes theorem
	
	while covered_probability < 0.99
		covered_probability = 1.0;
	endwhile
		
	c_p = covered_probability;							
endfunction

function [c_j_p_d, t_p] = compute_montecarlo_cjpd(g_b, t_m_a, s_m_a, o_v, u_v) 		#Montecarlo sampling approximation of the CPD
	state_card = columns(t_m_a(1).val);						
	observation_card = columns(s_m_a(1).val);
	t_h = length(o_v);
	total_probability = 0;
	
	c_j_p_d = 0; #TO BE IMPLEMENTED
		
	t_p = total_probability;
endfunction

function [c_j_p_d, t_p] = compute_marginalbased_cjpd(g_b, t_m_a, s_m_a, o_v, u_v) 	#Trajectory probabilities as product of marginals
	state_card = columns(t_m_a(1).val);						#BIG DISCLAIMER 2: this is conceptually WRONG
	observation_card = columns(s_m_a(1).val);					#is it even worth comparing?
	t_h = length(o_v);
	total_probability = 0;
														
	c_j_p_d = 0; #TO BE IMPLEMENTED
		
	t_p = total_probability;
endfunction

#################################################
#################################################
# RESILIENT PROPERTIES ##########################
#################################################
#################################################


function bool = l_resistant(l, t);			#true IFF a trajectory is l-resistant
	bool = true;
	for i=1:length(t)
		if t(1,i) > l
			bool = false;
		endif
	endfor
endfunction

function r_v = resistance_probability(max_l, c_j_p_d);				#sum the probabilities of resistent trajectories
	r_v = zeros(1, max_l);							#for a range of thresholds
	number_of_resistant_trajectories = zeros(1,max_l);
	for i=1:length(c_j_p_d)
		resistant_wrt_threshold = ones(1, max_l);
		for j=1:max_l
			if l_resistant(j, c_j_p_d(i).state_trajectory)		#alternative utility_trajectory
				r_v(1,j) += c_j_p_d(i).probability;
				number_of_resistant_trajectories(1,j)++;
			endif
		endfor
	endfor
endfunction

function print_resistance_vector_difference(r_v_1, r_v_2);		#compare two outputs of the previous function
	if length(r_v_1) != length(r_v_2)
		printf("ERROR: incompatible resistance vectors\n");
		exit;
	endif
	for i=1:length(r_v_1)
		printf("%d-resistance, real %f vs. %f, error %.0f%%\n", i, r_v_1(1,i), r_v_2(1,i), 100*abs(r_v_1(1,i)-r_v_2(1,i))/r_v_1(1,i));
	endfor
endfunction

function bool = p_q_recoverable(p, q, t);			#true IFF a trajectory is <p,q>-recoverable
	bool = true;
	extra_cost = 0;
	for i=1:length(t)
		if t(1,i) > p
			extra_cost += (t(1,i) - p);
		else 
			extra_cost = 0;
		endif
		if extra_cost > q
			bool = false;
		endif
	endfor	
endfunction

function r_m = recoverability_probability(max_p, max_q, c_j_p_d); 		#sum the probabilities of recoverable trajectories
	r_m = zeros(max_p, max_q);						#for the cartesian product of two ranges of parameters
	for i=1:max_p
		for j=1:max_q
			for k=1:length(c_j_p_d)
				if p_q_recoverable(i, j, c_j_p_d(k).state_trajectory)		#alternative utility_trajectory
					r_m(i,j) += c_j_p_d(k).probability;
				endif
			endfor
		endfor
	endfor	
endfunction

function print_recoverability_matrix_difference(r_m_1, r_m_2)			#compare two outputs of the previous function
	if (columns(r_m_1) != columns(r_m_2)) || (rows(r_m_1) != rows(r_m_2))
		printf("ERROR: incompatible recoverability matrices\n");
		exit;
	endif
	for i=1:rows(r_m_1)
		for j=1:columns(r_m_2)
			printf("<%d,%d>-recoverability, real %f vs %f, error %.0f%%\n", i, j, r_m_1(i,j), r_m_2(i,j), 100*(abs(r_m_1(i,j)-r_m_2(i,j))/r_m_1(i,j)));
		endfor
	endfor
endfunction

function bool = f_functional(f, t)				#true IFF a trajectory is f-functional
	bool = true;
	total = 0;
	for i=1:length(t)
		total += t(1,i);
	endfor
	if (total/length(t)) > f
		bool = false;
	endif
endfunction

function f_v = functionality_probability(max_f, c_j_p_d)			#sum the probabilities of functional trajectories
	f_v = zeros(1, max_f);							#for a range of thresholds
	number_of_functional_trajectories = zeros(1,max_f);
	for i=1:length(c_j_p_d)
		functional_wrt_threshold = ones(1, max_f);
		for j=1:max_f
			if f_functional(j, c_j_p_d(i).state_trajectory)		#alternative utility_trajectory
				f_v(1,j) += c_j_p_d(i).probability;
				number_of_functional_trajectories(1,j)++;
			endif
		endfor
	endfor
endfunction

function print_functionality_vector_difference(f_v_1, f_v_2);		#compare two outputs of the previous function
	if length(f_v_1) != length(f_v_2)
		printf("ERROR: incompatible functionality vectors\n");
		exit;
	endif
	for i=1:length(f_v_1)
		printf("%d-functionality, real %f vs. %f, error %.0f%%\n", i, f_v_1(1,i), f_v_2(1,i), 100*abs(f_v_1(1,i)-f_v_2(1,i))/f_v_1(1,i));
	endfor
endfunction

function bool = l_p_q_f_resilient(l, p, q, f, t)				#true IFF a trajectory is <l,p,q,f>-resilient
	if l_resistant(l,t) && p_q_recoverable(p,q,t) && f_functional(f,t)
		bool = true;
	else
		bool = false;
	endif
endfunction

function r_t = resilience_probability(max_l, max_p, max_q, max_f, c_j_p_d)	#sum the probabilities of resilient trajectories
	r_t = zeros(max_l*max_p*max_q*max_f,5);					#for the cartesian product of the parameters
	index = 1;
	for i=1:max_l
		for j=1:max_p
			for k=1:max_q
				for l=1:max_f
					r_t(index,:) = [i,j,k,l,0.0];
					for m=1:length(c_j_p_d)
						if l_p_q_f_resilient(i,j,k,l,c_j_p_d(m).state_trajectory)	
												#alternative utility_trajectory
							r_t(index,5) += c_j_p_d(m).probability;
						endif
					endfor
					index++;
				endfor
			endfor
		endfor
	endfor
endfunction

function print_resilience_table_difference(r_t_1, r_t_2)			#compare two outputs of the previous function
	if (columns(r_t_1) != columns(r_t_2)) || (rows(r_t_1) != rows(r_t_2))
		printf("ERROR: incompatible resilience tables\n");
		exit;
	endif
	for i=1:rows(r_t_1)
			if (r_t_1(i,1) != r_t_2(i,1)) || (r_t_1(i,2) != r_t_2(i,2)) || (r_t_1(i,3) != r_t_2(i,3)) || (r_t_1(i,4) != r_t_2(i,4))
				printf("WARNING: different resilience tables\n");
			endif
			printf("<%d,%d,%d,%d>-resilience, real %f vs %f, error %.0f%%\n", r_t_1(i,1), r_t_1(i,2), r_t_1(i,3), r_t_1(i,4), r_t_1(i,5), r_t_2(i,5), 100*(abs(r_t_1(i,5)-r_t_2(i,5))/r_t_1(i,5)));
	endfor
endfunction

#################################################
#################################################
# MAIN ##########################################
#################################################
#################################################

for iter=1:experiments
	printf("\nExperiment %d\n", iter);

	[ground_belief, transition_model_array, sensor_model_array, utility_vector] = choose_model(model_choice);
	disambiguated_observation_vector = observation_vector_disambiguation(observation_vector, sensor_model_array);

	integrity_checks(disambiguated_observation_vector, ground_belief, transition_model_array, sensor_model_array, utility_vector);

	[conditional_joint_probability_distribution, total_probability] = compute_cjpd(ground_belief, transition_model_array, sensor_model_array, disambiguated_observation_vector, utility_vector);

	sorted_conditional_joint_probability_distribution = sort_cjpd(conditional_joint_probability_distribution);
	printf("Actual CJPD\n");
	print_cjpd(sorted_conditional_joint_probability_distribution);

	heuristic_cjpd = compute_heuristic_cjpd(ground_belief, transition_model_array, sensor_model_array, disambiguated_observation_vector, utility_vector);
	sorted_heuristic_cjpd = sort_cjpd(heuristic_cjpd);
	printf("Heuristic CJPD\n");
	print_cjpd(sorted_heuristic_cjpd);
	
	#sorted_marginalbased_cjpd, sorted_montecarlo_cjpd							#TO BE IMPLEMENTED	
	
	total_absolute_error = compare_sorted_cjpds(sorted_conditional_joint_probability_distribution, sorted_heuristic_cjpd);
	printf("\nDistributions distance: %f\n", total_absolute_error);
	
	max_resistance_parameter = columns(transition_model_array(1).val);				#alternative: max(utility_vector);
	resistance_vector = resistance_probability(max_resistance_parameter, sorted_conditional_joint_probability_distribution);
	heuristic_resistance_vector = resistance_probability(max_resistance_parameter, sorted_heuristic_cjpd);
	print_resistance_vector_difference(resistance_vector, heuristic_resistance_vector);
	
	max_functionality_parameter = columns(transition_model_array(1).val);				#alternative: max(utility_vector);
	functionality_vector = resistance_probability(max_functionality_parameter, sorted_conditional_joint_probability_distribution);
	heuristic_functionality_vector = resistance_probability(max_functionality_parameter, sorted_heuristic_cjpd);
	print_functionality_vector_difference(functionality_vector, heuristic_functionality_vector);

	max_recoverability_p_parameter = columns(transition_model_array(1).val);			#alternative: max(utility_vector);
	max_recoverability_q_parameter = length(disambiguated_observation_vector)*max_recoverability_p_parameter;
	recoverability_matrix = recoverability_probability(max_recoverability_p_parameter, max_recoverability_q_parameter, sorted_conditional_joint_probability_distribution);
	heuristic_recoverability_matrix = recoverability_probability(max_recoverability_p_parameter, max_recoverability_q_parameter, sorted_heuristic_cjpd);
	print_recoverability_matrix_difference(recoverability_matrix, heuristic_recoverability_matrix);
	
	resilience_table = resilience_probability(max_resistance_parameter, max_recoverability_p_parameter, max_recoverability_q_parameter, max_functionality_parameter, sorted_conditional_joint_probability_distribution);
	heuristic_resilience_table = resilience_probability(max_resistance_parameter, max_recoverability_p_parameter, max_recoverability_q_parameter, max_functionality_parameter, sorted_heuristic_cjpd);
	print_resilience_table_difference(resilience_table, heuristic_resilience_table);
	
endfor
printf("\n");

