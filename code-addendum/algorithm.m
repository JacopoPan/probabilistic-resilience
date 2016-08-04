#1) P(obs|seq) from the sensor model
p_obs_given_seq = 1.0;
for j=1:(last_t)
	if j == 1 
		#initial ground belief prediction (and optional filtering)
		predicted_belief = initial_belief*transition_model;
		p_obs_given_seq *= sensor_model(encoded_state_trajectory(1),encoded_observations(1));
	else
		p_obs_given_seq *= sensor_model(encoded_state_trajectory(j),encoded_observations(j));
	endif
endfor
p_obs_given_seq;
#2) P(seq) from the transition model
p_seq = 0.0;
for j=1:(last_t)
	if j == 1 
		#initial ground belief prediction (and optional filtering)
		predicted_belief = initial_belief*transition_model;
		p_seq = predicted_belief(encoded_state_trajectory(1));
	else
		p_seq *= transition_model(encoded_state_trajectory(j-1),encoded_state_trajectory(j));
	endif
endfor
p_seq;
#3) P(obs) from the forward algorithm
for j=1:(last_t)
	if j == 1 
		#initial ground belief prediction (and optional filtering)
		predicted_forward = initial_belief*transition_model;
		filtered_forward = predicted_forward.*sensor_model(:,encoded_observations(1))';
		
	else
		predicted_forward = filtered_forward*transition_model;
		filtered_forward = predicted_forward.*sensor_model(:,encoded_observations(j))';
		
	endif
endfor
filtered_forward;
p_obs = sum(filtered_forward);
p_trajectory = (p_obs_given_seq*p_seq)/p_obs;