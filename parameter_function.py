# =============================================================================
# Filtering
# =============================================================================
def filter_parameter_sets_by_trend(result_list: list):
    """
    Analyzes model simulations to identify parameter sets that reproduce 
    the expected biological trend.
    
    Criteria for selection:
    1. The total error (P2 + P3) must decrease as drug concentration increases.
    2. State P2 must decrease.
    3. The decay of P2 must be faster (more sensitive) than the decay of P3.
    
    Returns:
        list: A filtered list of dictionaries containing parameters and their source model ID.
    """
    filtered_results = []

    # Use enumerate to track which model (by index) we are currently analyzing
    for model_id, model in enumerate(result_list):
        for parameter_set in model:
            # Extract the steady state distributions for each drug dose
            sim_data = parameter_set['results']
            
            # Lists to store local slopes (derivatives) between successive doses
            slopes_p2 = []
            slopes_p3 = []
            slopes_sum = []

            for i in range(len(sim_data) - 1):
                # Calculate the change in drug concentration (dx)
                dx = sim_data[i+1]['drug'] - sim_data[i]['drug']
                
                # Safety check: avoid division by zero or unordered data
                if dx <= 0: 
                    continue 

                # Calculate local slopes (y2 - y1) / dx
                m2 = (sim_data[i+1]['P2'] - sim_data[i]['P2']) / dx
                m3 = (sim_data[i+1]['P3'] - sim_data[i]['P3']) / dx
                
                # Calculate slope for the sum of incorrect states (P2 + P3)
                total_error_now = sim_data[i]['P2'] + sim_data[i]['P3']
                total_error_next = sim_data[i+1]['P2'] + sim_data[i+1]['P3']
                m_sum = (total_error_next - total_error_now) / dx

                slopes_p2.append(m2)
                slopes_p3.append(m3)
                slopes_sum.append(m_sum)

            # Proceed only if we have calculated slopes
            if not slopes_p2:
                continue

            # Calculate the average slope across the entire drug range
            avg_m2 = sum(slopes_p2) / len(slopes_p2)
            avg_m3 = sum(slopes_p3) / len(slopes_p3)
            avg_m_sum = sum(slopes_sum) / len(slopes_sum)

            # --- TREND FILTERING LOGIC ---
            # 1. P2 + P3 must be decreasing (avg_m_sum < 0)
            # 2. P2 must be decreasing (avg_m2 < 0)
            # 3. P2 must drop more sharply than P3 (|m2| > |m3|)
            if avg_m_sum < 0 and avg_m2 < 0:
                if abs(avg_m2) > abs(avg_m3):
                    # Store a dictionary with the parameters and the source model ID
                    # Using .copy() ensures we don't accidentally modify the original data
                    tag_params = parameter_set['params'].copy()
                    tag_params['origin_model'] = f"Model {model_id + 1}"
                    filtered_results.append(tag_params)
        
    return filtered_results

