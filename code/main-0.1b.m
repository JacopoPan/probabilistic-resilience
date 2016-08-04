#! /usr/bin/octave -qf
# note: the interpreter path is /usr/bin/octave in ubuntu, /usr/local/bin/octave in mac osx

#model specification

##{
#rain model
ground_belief = [0.5, 0.5]; #P(rain, no rain)
transition_set(1).transition_model = [0.8, 0.2; #P(rain, no rain|it was raining)
		    0.2, 0.8]; #P(rain, no rain|it wasn't raining)
sensor_set(1).sensor_model = [0.8, 0.2; #P(umbrella, no umbrella|it is raining)
		0.1, 0.9]; #P(umbrella, no umbrella|it isn't raining)
#}

#{
#sickness model
ground_belief = [0.33, 0.33, 0.33]; #P(healthy, sick, very sick)
transition_set(1).transition_model = [0.8, 0.15, 0.05; #P(healthy, sick, very sick|was healthy)
		    0.2, 0.7, 0.1; #P(healthy, sick, very sick|was sick)
		    0.1, 0.3, 0.6]; #P(healthy, sick, very sick|was very sick)
sensor_set(1).sensor_model = [0.9, 0.05, 0.05; #P(normal, warm, very warm|is healthy)
		0.2, 0.7, 0.1; #P(normal, warm, very warm|is sick)
		0.1, 0.6, 0.3]; #P(normal, warm, very warm|is very sick)
#}

#querying the model
query_at_time_step = 1; #the time step at which we compute the marginal of the hidden variable
observations = "22"; #values refer to a column in the sensor model, use "-" for <no observation>
#the length of the time horizon is the maximum among "query_at_time_step" and "length(observations)"
#if "query_at_time_step" > "length(observations)" we assume <no observation> ("-") in the interval [length(observations), query_at_time_step]
produce_inference = 1; #boolean variable saying to execute the inference algorithms block
compare_with_ground_truth = 1; #boolean variable saying to compute the full join distribution for reference









#parameters computation
last_o = length(observations);
last_t = max(query_at_time_step, length(observations));
if query_at_time_step > length(observations)
	for i=1:(query_at_time_step-length(observations)) observations = strcat(observations, "-"); endfor
endif
state_cardinality = columns(transition_set(1).transition_model);
observation_cardinality = columns(sensor_set(1).sensor_model);

#integrity checks
if rows(transition_set(1).transition_model) != state_cardinality printf("inconsistent transition model.\n"); return; endif
if rows(sensor_set(1).sensor_model) != state_cardinality printf("inconsistent sensor model.\n"); return; endif
if query_at_time_step > last_t printf("you are querying too far.\n"); return; endif
for i=1:length(observations)
	[num, bool] = str2num(substr(observations,i,1));
	if bool == 0 && !(strcmp(substr(observations,i,1),"-"))
		printf("impossible observations.\n");
		return;
	elseif (num > observation_cardinality) || (num < 1)
		printf("impossible observations.\n");
		return;
	endif
endfor









#print out model representation
printf("\n####################\nmodel representation\n####################\n\n");
#ground belief
printf("(");
for i=1:(state_cardinality-1)
	printf("%1.2f,", ground_belief(i));
endfor
printf("%1.2f) -> ", ground_belief(state_cardinality));
#hidden variables until the last observation
for i=1:last_o
	printf("%d:[1-%d] -> ", i, state_cardinality);
endfor
#possible hidden variable at query step
if query_at_time_step > last_o
	if query_at_time_step == (last_o+1)
		printf("%d:[1-%d] -> ...\n", query_at_time_step, state_cardinality);
	else
		printf("... -> %d:[1-%d] -> ...\n", query_at_time_step, state_cardinality);
	endif
else
	printf("...\n");
endif

#first parts of the arrows
printf("         ");
for i=1:(state_cardinality) printf("     "); endfor
for i=1:last_o printf("|          "); endfor
if (query_at_time_step > last_o)
	if query_at_time_step == (last_o+1)
		printf("|\n", query_at_time_step, state_cardinality);
	else
		printf("       |\n", query_at_time_step, state_cardinality);
	endif
else
	printf("\n");
endif

#second parts of the arrows
printf("         ");
for i=1:(state_cardinality) printf("     "); endfor
for i=1:last_o printf("v          "); endfor
if (query_at_time_step > last_o)
	if query_at_time_step == (last_o+1)
		printf("v\n", query_at_time_step, state_cardinality);
	else
		printf("       v\n", query_at_time_step, state_cardinality);
	endif
else
	printf("\n");
endif

#observations
printf("         ");
for i=1:(state_cardinality) printf("     "); endfor
for i=1:last_o printf("%s          ", observations(i)); endfor
if (query_at_time_step > last_o)
	if query_at_time_step == (last_o+1)
		printf("-\n", query_at_time_step, state_cardinality);
	else
		printf("       -\n", query_at_time_step, state_cardinality);
	endif
else
	printf("\n");
endif










#helper function to print out trajectories and observations
function retval = visual_string(traj_obs_string)
	retval = "";
	if length(traj_obs_string) == 0 
		printf("warning: trying to visually print out an empty string\n");
	else
		for i=1:(length(traj_obs_string)-1)
			retval = strcat(retval, substr(traj_obs_string,i,1));
			retval = strcat(retval, ", ");
		endfor
		retval = strcat(retval, substr(traj_obs_string,length(traj_obs_string),1));
	endif
endfunction









#helper function to compare observation strings
function retval = compatible_traj(this_traj, template_traj)
	retval = 1;
	if length(this_traj) != length(template_traj)
		printf("strings of different length\n");
		retval = 0;
		break;
	endif
	for i=1:length(this_traj)
		if !strcmp(substr(this_traj,i,1),substr(template_traj,i,1)) && !strcmp(substr(template_traj,i,1),"-")
			retval = 0;
			break;
		endif
	endfor
endfunction









#helper function to compute the utility of the hidden variable
function retval = utility(hv_value)
	retval = hv_value*hv_value; #e.g. a quadratic function
endfunction






#helper function to compute the utility of a trajectory
function retval = trajectory_utility(trajectory)
	retval = 0;
	for i=1:length(trajectory)
		[num, bool] = str2num(substr(trajectory,i,1))
		if bool == 1
			retval += utility(num);
		else
			prinf("error in parsing the trajectory for utility computation!\n");
		endif
	endfor
endfunction









#helper function to compute the utility of a trajectory
function retval = trajectory_property(trajectory, property_name, parameters)
	retval = 0;
	switch property_name
		case "resistance"

		case "recoverability"

		case "functionality"

		otherwise

	endswitch
endfunction








if produce_inference == 1
#predict, filter, smooth
#note: the same code is replicated for: 1) filtering with missing observations, 2) prediction with some observations, and 3) the forward factor in the smoothing algorithm
#the only practical difference is the computation of the backward factor in the smoothing algorithm
if (query_at_time_step-last_o) == 0
	printf("\n#####################################################################\n[filtered] marginal distribution of the hidden variable at timestep %d\n#####################################################################\n\n", query_at_time_step);
	#initial ground belief prediction (and optional filtering)
	predicted_belief = ground_belief*transition_set(1).transition_model;
	[current_state, bool] = str2num(substr(observations,1,1));
	if bool == 1
		filtered_belief = predicted_belief.*sensor_set(1).sensor_model(:,current_state)';
		filtered_belief = filtered_belief./sum(filtered_belief);
	else
		filtered_belief = predicted_belief; #???
	endif
	#propagate belief to deal with queries at subsequent time steps
	if query_at_time_step > 1
		for i=2:query_at_time_step
			predicted_belief = filtered_belief*transition_set(1).transition_model;
			[current_state, bool] = str2num(substr(observations,i,1));
			#optional filtering
			if bool == 1
				filtered_belief = predicted_belief.*sensor_set(1).sensor_model(:,current_state)';
				filtered_belief = filtered_belief./sum(filtered_belief);
			else
				filtered_belief = predicted_belief; #???
			endif
		endfor
	endif
	#print the result of query
	for i=1:state_cardinality
		printf("P(x_{%d}=%d|o_{1:%d}=[%s])\t", query_at_time_step, i, last_t, visual_string(observations));
	endfor
	printf("\n> ");
	for i=1:state_cardinality
		printf("%f\t\t\t", filtered_belief(i));
	endfor
	printf("\n");
		
elseif (query_at_time_step-last_o) > 0
	printf("\n######################################################################\n[predicted] marginal distribution of the hidden variable at timestep %d\n######################################################################\n\n", query_at_time_step);
	#initial ground belief prediction (and optional filtering)
	predicted_belief = ground_belief*transition_set(1).transition_model;
	[current_state, bool] = str2num(substr(observations,1,1));
	if bool == 1
		filtered_belief = predicted_belief.*sensor_set(1).sensor_model(:,current_state)';
		filtered_belief = filtered_belief./sum(filtered_belief);
	else
		filtered_belief = predicted_belief; #???
	endif
	#propagate belief to deal with queries at subsequent time steps
	if query_at_time_step > 1
		for i=2:query_at_time_step
			predicted_belief = filtered_belief*transition_set(1).transition_model;
			[current_state, bool] = str2num(substr(observations,i,1));
			#optional filtering
			if bool == 1
				filtered_belief = predicted_belief.*sensor_set(1).sensor_model(:,current_state)';
				filtered_belief = filtered_belief./sum(filtered_belief);
			else
				filtered_belief = predicted_belief; #???
			endif
		endfor
	endif
	#print the result of query
	for i=1:state_cardinality
		printf("P(x_{%d}=%d|o_{1:%d}=[%s])\t", query_at_time_step, i, last_t, visual_string(observations));
	endfor
	printf("\n> ");
	for i=1:state_cardinality
		printf("%f\t\t\t", filtered_belief(i));
	endfor
	printf("\n");
	
elseif (query_at_time_step-last_o) < 0
	printf("\n#####################################################################\n[smoothed] marginal distribution of the hidden variable at timestep %d\n#####################################################################\n\n", query_at_time_step);
	#initial ground belief prediction (and optional filtering)
	predicted_belief = ground_belief*transition_set(1).transition_model;
	[current_state, bool] = str2num(substr(observations,1,1));
	if bool == 1
		filtered_belief = predicted_belief.*sensor_set(1).sensor_model(:,current_state)';
		filtered_belief = filtered_belief./sum(filtered_belief);
	else
		filtered_belief = predicted_belief; #???
	endif
	forward = filtered_belief;
	#propagate belief to deal with queries at subsequent time steps
	if query_at_time_step > 1
		for i=2:query_at_time_step
			predicted_belief = filtered_belief*transition_set(1).transition_model;
			[current_state, bool] = str2num(substr(observations,i,1));
			#optional filtering
			if bool == 1
				filtered_belief = predicted_belief.*sensor_set(1).sensor_model(:,current_state)';
				filtered_belief = filtered_belief./sum(filtered_belief);
			else
				filtered_belief = predicted_belief; #???
			endif
		endfor
		forward = filtered_belief;
	endif
	#compute the backward factor, i.e. P(o_{query_at_time_step+1:last_o}|filtered_belief)
	t = last_o; #just the renaming of a constant
	k = query_at_time_step; #just the renaming of a constant
	backward_k = t; #backward index
	for i=1:(t-k+1)
		if backward_k == t 
			backward = ones(1, state_cardinality);
		else 
			[num_obs, bool] = str2num(substr(observations, backward_k+1, 1));
			new_backward = zeros(1, state_cardinality);
			if bool == 0
				printf("warning, smoothing with missing observations!\n"); #???
				for j=1:state_cardinality
					new_backward += backward(1,j).*transition_set(1).transition_model(:,j)';
				endfor
			else
				for j=1:state_cardinality
					new_backward += sensor_set(1).sensor_model(j,num_obs).*backward(1,j).*transition_set(1).transition_model(:,j)';
				endfor
			endif	
			backward = new_backward;		
		endif	
		backward_k--;	
	endfor
	smoothed_belief = forward.*backward;
	smoothed_belief = smoothed_belief./sum(smoothed_belief);
	#print the result of query
	for i=1:state_cardinality
		printf("P(x_{%d}=%d|o_{1:%d}=[%s])\t", query_at_time_step, i, last_t, visual_string(observations));
	endfor
	printf("\n> ");
	for i=1:state_cardinality
		printf("%f\t\t\t", smoothed_belief(i));
	endfor
	printf("\n");
	
else
	printf("\nerror: no marginal algorithm\n\n");
	return;
endif









#most likely sequence
printf("\n######################\nmost likely trajectory\n######################\n\n");
ml_trajectory = "";
for i=1:(last_t)
	if i == 1 
		#initial ground belief prediction (and optional filtering)
		predicted_belief = ground_belief*transition_set(1).transition_model;
		[current_state, bool] = str2num(substr(observations,1,1));
		if bool == 1
			filtered_belief = predicted_belief.*sensor_set(1).sensor_model(:,current_state)';
			filtered_belief = filtered_belief./sum(filtered_belief);
		else
			filtered_belief = predicted_belief; #???
		endif
		#rename
		viterbi_update = filtered_belief;
		#choose maximum
		[maximum, index] = max(viterbi_update);
		ml_trajectory = strcat(ml_trajectory, num2str(index)); #build most likely sequence
	else
		[current_obs, bool] = str2num(substr(observations,i,1));
		if bool == 1
			viterbi_update = sensor_set(1).sensor_model(:,current_obs)'.*transition_set(1).transition_model(index,:).*maximum;
		else
			viterbi_update = transition_set(1).transition_model(index,:).*maximum; #???
		endif
		#choose maximum
		[maximum, index] = max(viterbi_update);
		ml_trajectory = strcat(ml_trajectory, num2str(index)); #build most likely sequence
	endif
endfor
ml_trajectory = "11";
#compute probability of the ml sequence given the observations P(seq|obs) as (P(obs|seq)*P(seq))/P(obs)
#1) P(obs|seq) from the sensor model
p_obs_given_seq = 1.0;
for i=1:(last_t)
	if i == 1 
		#initial ground belief prediction (and optional filtering)
		predicted_belief = ground_belief*transition_set(1).transition_model;
		[current_hv, bool1] = str2num(substr(ml_trajectory,1,1));
		[current_ob, bool2] = str2num(substr(observations,1,1));
		if (bool1 == 1) && (bool2 == 1)
			p_obs_given_seq *= sensor_set(1).sensor_model(current_hv,current_ob);
		elseif bool2 == 0
			p_obs_given_seq *= 1.0;
		else
			printf("error in computing p_seq_given_seq, non-numerical trajectory!\n");
		endif
	else
		[current_hv, bool1] = str2num(substr(ml_trajectory,i,1));
		[current_ob, bool2] = str2num(substr(observations,i,1));
		if (bool1 == 1) && (bool2 == 1)
			p_obs_given_seq *= sensor_set(1).sensor_model(current_hv,current_ob);
		elseif bool2 == 0
			p_obs_given_seq *= 1.0;
		else
			printf("error in computing p_seq_given_seq, non-numerical trajectory!\n");
		endif
	endif
endfor
p_obs_given_seq
#2) P(seq) from the transition model
p_seq = 0.0;
for i=1:(last_t)
	if i == 1 
		#initial ground belief prediction (and optional filtering)
		predicted_belief = ground_belief*transition_set(1).transition_model;
		[current_hv, bool] = str2num(substr(ml_trajectory,1,1));
		if bool == 1
			p_seq = predicted_belief(current_hv);
			last_hv = current_hv;
		else
			printf("error in computing p_seq, non-numerical trajectory!\n");
		endif
		
	else
		[current_hv, bool] = str2num(substr(ml_trajectory,i,1));
		if bool == 1
			p_seq *= transition_set(1).transition_model(last_hv,current_hv);
			last_hv = current_hv;
		else
			printf("error in computing p_seq, non-numerical trajectory!\n");
		endif
		
	endif
endfor
p_seq
#3) P(obs) from the forward algorithm
for i=1:(last_t)
	if i == 1 
		#initial ground belief prediction (and optional filtering)
		predicted_forward = ground_belief*transition_set(1).transition_model;
		[current_ob, bool] = str2num(substr(observations,1,1));
		if (bool == 1)
			#note: NO NORMALIZATION HERE
			filtered_forward = predicted_forward.*sensor_set(1).sensor_model(:,current_ob)';
		else
			filtered_forward = predicted_forward;
			#printf("warning: in computing p_obs, missing observations!\n");
		endif
		
	else
		predicted_forward = filtered_forward*transition_set(1).transition_model;
		[current_ob, bool] = str2num(substr(observations,i,1));
		if (bool == 1)
			#note: NO NORMALIZATION HERE
			filtered_forward = predicted_forward.*sensor_set(1).sensor_model(:,current_ob)';
		else
			filtered_forward = predicted_forward;
			#printf("warning: in computing p_obs, missing observations!\n");
		endif
		
	endif
endfor
filtered_forward
p_obs = sum(filtered_forward)
p_trajectory = (p_obs_given_seq*p_seq)/p_obs;
printf("- P(x_{1:%d}=[%s]|o_{1:%d}=[%s]) = %f\n", last_t, visual_string(ml_trajectory), last_t, visual_string(observations), p_trajectory);

endif #inference block









if compare_with_ground_truth == 1
#compute ground truth table

#(1) build trajectories recursively
function retval = build_trajectories_from_to(t_1, last_t, state_cardinality)
	if t_1 == last_t
		for j=1:state_cardinality
			retval(j).traj = num2str(j);
		endfor
	else
		for j=1:state_cardinality
			shallow_list(j).traj = num2str(j);
		endfor
		deeper_list = build_trajectories_from_to(t_1+1, last_t, state_cardinality);
		for j=1:length(shallow_list)
			for k=1:length(deeper_list)
				#cartesian product of the two lists
				retval(length(deeper_list)*(j-1)+k).traj = strcat(shallow_list(j).traj,deeper_list(k).traj);
			endfor
		endfor
	endif
endfunction
trajectories_list = build_trajectories_from_to(1, last_t, state_cardinality);

#(2) build observations recursively
function retval = build_observations_from_to(t_1, last_t, observation_cardinality)
	if t_1 == last_t
		for j=1:observation_cardinality
			retval(j).obs = num2str(j);
		endfor
	else
		for j=1:observation_cardinality
			shallow_list(j).obs = num2str(j);
		endfor
		deeper_list = build_observations_from_to(t_1+1, last_t, observation_cardinality);
		for j=1:length(shallow_list)
			for k=1:length(deeper_list)
				#cartesian product of the two lists
				retval(length(deeper_list)*(j-1)+k).obs = strcat(shallow_list(j).obs,deeper_list(k).obs);
			endfor
		endfor
	endif
endfunction
observations_list = build_observations_from_to(1, last_t, observation_cardinality);

#(3) build full table
entry = 1;
for i=1:state_cardinality^last_t
	for j=1:observation_cardinality^last_t
		full_table(entry).traj = trajectories_list(i).traj;
		full_table(entry).obs = observations_list(j).obs;
		entry++;
	endfor
endfor

#(4) compute probabilities
printf("\n##########################\nunrolled join distribution\n##########################\n\n");
for i=1:length(full_table)
	#probability of a trajectory no matter the observations
	for j=1:last_t
		current_state = str2num(substr(full_table(i).traj, j, 1));
		if j == 1
			full_table(i).prob = ground_belief*transition_set(1).transition_model(:,current_state);;
		else
			last_state = str2num(substr(full_table(i).traj, j-1, 1));
			full_table(i).prob = full_table(i).prob*transition_set(1).transition_model(last_state,current_state);
		endif
	endfor	
	#multiplication by the probability of producing exactly those observations in the given trajectory
	for j=1:last_t
		current_state = str2num(substr(full_table(i).traj, j, 1));
		current_obs = str2num(substr(full_table(i).obs, j, 1));
		full_table(i).prob = full_table(i).prob*sensor_set(1).sensor_model(current_state,current_obs);
	endfor
	printf(". P(x_{1:%d}=[%s], o_{1:%d}=[%s]) = %f\n", last_t, visual_string(full_table(i).traj), last_t, visual_string(full_table(i).obs), full_table(i).prob);
endfor

#check
P = 0; for i=1:length(full_table) P += full_table(i).prob; endfor 
#P









#marginals of the hidden variable
hv_distr = zeros(1,state_cardinality);
for i=1:length(full_table)
	for j = 1:state_cardinality
		if (str2num(substr(full_table(i).traj, query_at_time_step, 1)) == j) && (compatible_traj(full_table(i).obs,observations))
			hv_distr(j) += full_table(i).prob;
		endif
	endfor
endfor
hv_distr = hv_distr./sum(hv_distr);
printf("\n##########################################################\nmarginal distribution of the hidden variable at timestep %d\n##########################################################\n\n> ", query_at_time_step);
for i=1:state_cardinality
	printf("P(x_{%d}=%d|o_{1:%d}=[%s])\t", query_at_time_step, i, last_t, visual_string(observations));
endfor
printf("\n> ");
for i=1:state_cardinality
	printf("%f\t\t\t", hv_distr(i));
endfor
printf("\n");

#trajectories
entry = 1;
normalization = 0;
for i=1:length(full_table)
	if compatible_traj(full_table(i).obs,observations)
		new_entry = 1;
		if entry > 1
			for j=1:length(pruned_table)
				if strcmp(pruned_table(j).traj, full_table(i).traj)
					pruned_table(j).prob = pruned_table(j).prob + full_table(i).prob;
					new_entry = 0;
				endif
			endfor
		endif
		if new_entry == 1;
			pruned_table(entry).traj = full_table(i).traj;
			pruned_table(entry).prob = full_table(i).prob;
			entry++;
		endif
		normalization += full_table(i).prob;
	endif
endfor
printf("\n##############################################################\ndistribution over all the trajectories, given the observations\n##############################################################\n\n");
for i=1:length(pruned_table)
	pruned_table(i).prob = (pruned_table(i).prob)/normalization;
	printf("- P(x_{1:%d}=[%s]|o_{1:%d}=[%s]) = %f\n", last_t, visual_string(pruned_table(i).traj), last_t, visual_string(observations), pruned_table(i).prob);
endfor
printf("\n");
#check
P = 0; for i=1:length(pruned_table) P += pruned_table(i).prob; endfor 
#P

endif #ground truth block
