# Next steps

# To remember

Name										Meaning												Shape
window_bits							raw input window							[1, input_bits]
input_layer_output			input layer output						[1, N_in]
state_bits							recurrent state								[1, N_state]
state_layer_input				[input_out(t), state(t-1)]		[1, N_in + N_state]
state_layer_output			state(t)											[1, N_state]
output_layer_input			[input_out(t), state(t)]			[1, N_in + N_state]
output_layer_output			final output									[1, N_out]
