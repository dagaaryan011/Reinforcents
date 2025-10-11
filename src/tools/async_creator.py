import random

def async_batch_creator(insti_agents: list, retail_agents: list, mm_agents: list):
    #making random batches to make a psuedo async environment 
    insti_no = len(insti_agents)
    retail_no = len(retail_agents)
    mm_no = len(mm_agents)

    b_1_mm = random.randint(0, mm_no)
    b_2_mm = random.randint(0, mm_no - b_1_mm)
    b_3_mm = mm_no - b_1_mm - b_2_mm

    b_1_retail = random.randint(0, retail_no)
    b_2_retail = random.randint(0, retail_no - b_1_retail)
    b_3_retail = retail_no - b_1_retail - b_2_retail

    b_1_insti = random.randint(0, insti_no)
    b_2_insti = random.randint(0, insti_no - b_1_insti)
    b_3_insti = insti_no - b_1_insti - b_2_insti

    indices_insti = list(range(insti_no))
    indices_mm = list(range(mm_no))
    indices_retail = list(range(retail_no))
    
    random.shuffle(indices_insti)
    random.shuffle(indices_mm)
    random.shuffle(indices_retail)

    insti_b1 = [insti_agents[i] for i in indices_insti[0 : b_1_insti]]
    insti_b2 = [insti_agents[i] for i in indices_insti[b_1_insti : b_1_insti + b_2_insti]]
    insti_b3 = [insti_agents[i] for i in indices_insti[b_1_insti + b_2_insti :]]

    retail_b1 = [retail_agents[i] for i in indices_retail[0 : b_1_retail]]
    retail_b2 = [retail_agents[i] for i in indices_retail[b_1_retail : b_1_retail + b_2_retail]]
    retail_b3 = [retail_agents[i] for i in indices_retail[b_1_retail + b_2_retail :]]

    mm_b1 = [mm_agents[i] for i in indices_mm[0 : b_1_mm]]
    mm_b2 = [mm_agents[i] for i in indices_mm[b_1_mm : b_1_mm + b_2_mm]]
    mm_b3 = [mm_agents[i] for i in indices_mm[b_1_mm + b_2_mm :]]

    batch1 = [mm_b1, insti_b1, retail_b1]
    batch2 = [mm_b2, insti_b2, retail_b2]
    batch3 = [mm_b3, insti_b3, retail_b3]

    return batch1, batch2, batch3