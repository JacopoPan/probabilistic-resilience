#! /usr/bin/octave -qf
#Ubuntu's interpreter /usr/bin/octave
#OSX's interpreter /usr/local/bin/octave

#################################################
#################################################
# PARAMETERS ####################################
#################################################
#################################################

model_choice = "sick";		#"rain" for the rain model, "sick" for the sickness model, "random" for autogeneration
observation_vector = [1,1];	#integers referring to columns in the sensor model, use "0" for <missing>, or "random" for autogeneration

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


#################################################
#################################################
# AD HOC ALGORITHMS FOR RESILIENT PROPERTIES ####
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
	number_of_resistant_trajectories
endfunction

function print_resistance_vector(r_v);
	for i=1:length(r_v)
		printf("%d-resistance, %f\n", i, r_v(1,i));
	endfor
endfunction

function p = compute_l_resistance(g_b, t_m_a, s_m_a, o_v, u_v, l) 
	state_card = columns(t_m_a(1).val);
	observation_card = columns(s_m_a(1).val);
	t_h = length(o_v);
	
	#reorder gb, tm, sm, ov by uv  (optional)
	
	#aggregate gb, tm, sm, ov by l
	index = 1;
	aggregated_g_b = cat(2,  sum(g_b(:,1:index),2)  ,  sum(g_b(:,index+1:columns(g_b)),2)  );
	
	
	temp = cat(2,  sum(t_m_a(1).val(:,1:index),2)  ,  sum(t_m_a(1).val(:,index+1:columns(t_m_a(1).val)),2)  );
	aggregated_t_m = cat(1,  (sum(temp(1:index,:),1))./index  ,  (sum(temp(index+1:rows(temp),:),1))./(rows(temp)-index)  )
	
	temp = cat(2,  sum(s_m_a(1).val(:,1:index),2)  ,  sum(s_m_a(1).val(:,index+1:columns(s_m_a(1).val)),2)  );
	aggregated_s_m = cat(1,  sum(temp(1:index,:),1)./index  ,  sum(temp(index+1:rows(temp),:),1)./(rows(temp)-index)  )
		
	aggregated_o_v = ones(1, length(o_v));
	for i = 1:length(aggregated_o_v)
		if o_v(1,i) > index
			aggregated_o_v = 2;
		endif
	endfor
	aggregated_o_v;
	
	#compute probability of the "less than l" (1) trajectory with the given observations
	for i = 1:t_h
		if i == 1
			p = aggregated_g_b(1,1)*aggregated_t_m(1,1)+aggregated_g_b(1,2)*aggregated_t_m(2,1);
			p *= aggregated_s_m(1,aggregated_o_v(1,i));
		else
			p *= aggregated_t_m(1,1);
			p *= aggregated_s_m(1,aggregated_o_v(1,i));
		endif
	endfor
	p;
	
	#aggregated_g_b = g_b;
	#aggregated_t_m = t_m_a(1).val;
	#aggregated_s_m  = s_m_a(1).val;
	
	for i=1:(t_h)
		if i == 1 
			#initial ground belief prediction (and optional filtering)
			predicted_forward = aggregated_g_b*aggregated_t_m;
			filtered_forward = predicted_forward.*aggregated_s_m(:,aggregated_o_v(1,i))';
		else
			predicted_forward = filtered_forward*aggregated_t_m;
			filtered_forward = predicted_forward.*aggregated_s_m(:,aggregated_o_v(1,i))';
		endif
	endfor
	p_obs = sum(filtered_forward) #0.27875
	real = 0.27875
					
	p = p/p_obs;					
endfunction



#################################################
#################################################
# MAIN ##########################################
#################################################
#################################################
printf("\n");

[ground_belief, transition_model_array, sensor_model_array, utility_vector] = choose_model(model_choice);
disambiguated_observation_vector = observation_vector_disambiguation(observation_vector, sensor_model_array);

integrity_checks(disambiguated_observation_vector, ground_belief, transition_model_array, sensor_model_array, utility_vector);

[conditional_joint_probability_distribution, total_probability] = compute_cjpd(ground_belief, transition_model_array, sensor_model_array, disambiguated_observation_vector, utility_vector);

sorted_conditional_joint_probability_distribution = sort_cjpd(conditional_joint_probability_distribution);
printf("CJPD\n");
print_cjpd(sorted_conditional_joint_probability_distribution);

printf("\n");
max_resistance_parameter = columns(transition_model_array(1).val);				#alternative: max(utility_vector);
resistance_vector = resistance_probability(max_resistance_parameter, sorted_conditional_joint_probability_distribution);
print_resistance_vector(resistance_vector);




	printf("\n"); l=2;
	#compute l resistance at step 1
	s1_before_o = ground_belief*transition_model_array.val;
	temp = s1_before_o.*sensor_model_array.val(:,observation_vector(1))';
	s1_after_o = temp./sum(temp)
	#backward factor
	t = 2; k = 1; #just the renaming of constants
	backward_k = t; #backward index
	for i=1:(t-k+1)
		if backward_k == t 
			backward = ones(1, 3);
		else 
			#[num_obs, bool] = str2num(substr(observations, backward_k+1, 1));
			obs = observation_vector(backward_k+1);
			new_backward = zeros(1, 3);
			for j=1:3
				new_backward += sensor_model_array.val(j,obs).*backward(1,j).*transition_model_array.val(:,j)';
			#new_backward += sensor_set(1).sensor_model(j,num_obs).*backward(1,j).*transition_set(1).transition_model(:,j)';
			endfor
			backward = new_backward;		
		endif	
		backward_k--;	
	endfor
	smoothed_belief = s1_after_o.*backward;
	smoothed_belief = smoothed_belief./sum(smoothed_belief)
	p_of_l_resistance_1 = sum(smoothed_belief(1:l)) #BE CAREFUL OF THE ROUNDING UP!!!
	
	
	
	#zeros non resistant belief and transitions
	#s1_after_o(1,3) = 0.0;
	#transition_model_array.val(1,3) = 0.0; transition_model_array.val(2,3) = 0.0;
	#transition_model_array.val(3,1) = 0.0; transition_model_array.val(3,2) = 0.0; #transition_model_array.val(3,3) = 0.0;


	#compute l resistance at step 2
	s2_before_o = smoothed_belief*transition_model_array.val
	s2_before_o = s2_before_o./sum(s2_before_o)
	temp = s2_before_o.*sensor_model_array.val(:,observation_vector(2))';
	s2_after_o = temp./sum(temp)
	#backward factor
	t = 2; k = 2; #just the renaming of constants
	backward_k = t; #backward index
	for i=1:(t-k+1)
		if backward_k == t 
			backward = ones(1, 3);
		else 
			#[num_obs, bool] = str2num(substr(observations, backward_k+1, 1));
			obs = observation_vector(backward_k+1);
			new_backward = zeros(1, 3);
			for j=1:3
				new_backward += sensor_model_array.val(j,obs).*backward(1,j).*transition_model_array.val(:,j)';
			#new_backward += sensor_set(1).sensor_model(j,num_obs).*backward(1,j).*transition_set(1).transition_model(:,j)';
			endfor
			backward = new_backward;		
		endif	
		backward_k--;	
	endfor
	smoothed_belief2 = s2_after_o.*backward;
	smoothed_belief2 = smoothed_belief2./sum(smoothed_belief2)
	p_of_l_resistance_2 = sum(s2_after_o(1:l)) #BE CAREFUL OF THE ROUNDING UP!!!



	smoothed_belief
	smoothed_belief2
	p_of_l_resistance_2*p_of_l_resistance_1
	
	
	
	
printf("\n");

